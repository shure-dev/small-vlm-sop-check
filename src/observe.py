"""SOP定義の questions: からプロンプトを自動生成し、
ローカル小型VLM(Qwen3-VL, mlx_vlm)でフレームごとに観察する（Phase 1）。

各質問への回答だけでなく、生成トークンのlogitから実測した信頼度(自己申告ではない)も
一緒に返す。これによりPhase 2(judge)側で「低信頼な観察に頼った判定か」を
可視化できる(experiments/sop_step_detect/confidence_judge/ での実験で技術検証済み)。

ドメイン固有の知識(ガスコンロの点検作業など)は一切持たない。
SOP定義ファイルの `questions:` セクションだけを見てプロンプトを組み立てる。
"""
from __future__ import annotations
import math
import re
import time
from typing import Any


def _as_yaml_safe_str(v: Any) -> str:
    """YAMLは 'yes'/'no' のようなクォート無し語をブール値(True/False)と解釈してしまう
    (YAML 1.1のブール語彙)。questions[].values にこの罠があっても壊れないよう、
    ブール値なら"yes"/"no"へ戻し、それ以外は素直にstr化する。
    SOPファイル側は values に必ずクォート付き文字列("yes"等)を書くのが正しい書き方だが、
    ここでは防御的に吸収する。
    """
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def build_prompt(questions: list[dict[str, Any]], domain_hint: str, t: float) -> str:
    """SOPの questions: 定義から、1フレーム分の観察プロンプトを自動生成する。

    設計(experiments/ で7モデル実測):
    - 質問文は値スロットに入れず legend に分離する。値スロットに質問文を入れると
      MiniCPM-V等の小型モデルが値へ質問文をそのままエコーし、yes/noが出なくなる。
    - 指示は英語にすると小型モデルでも追従しやすい(質問文自体は元の言語のまま保持)。
    - prefill='{"'(既定)と併用する前提。空の値スロットだけだと Molmo が「完成済み」と
      みなして空応答するが、prefillが最初のキーの途中まで固定するので生成が続く。
    """
    legend_parts, schema_parts = [], []
    for c in questions:
        values = [_as_yaml_safe_str(v) for v in c.get("values", ["yes", "no"])]
        legend_parts.append(f'- {c["id"]}: {c["ask"]} (answer with {" or ".join(values)})')
        schema_parts.append(f'"{c["id"]}":""')
    legend = "\n".join(legend_parts)
    schema = "{" + ",".join(schema_parts) + "}"
    return (
        f"{domain_hint} (time t={t}s). Report only what you can see (no guessing).\n"
        f"Fields:\n{legend}\n"
        "Fill each JSON value with exactly one allowed word (e.g. yes/no/unclear). "
        "Do NOT repeat the question text as the value:\n" + schema
    )


class Observer:
    """VLMをロードして、フレーム1枚ごとの観察+信頼度を返すオブジェクト。

    使い方:
        obs = Observer(model="mlx-community/Qwen3-VL-4B-Instruct-4bit", questions=sop["questions"])
        record = obs.ask(image_path, t=7.0, domain_hint=sop["sop"]["domain_hint"])
        # record = {"raw": "...", "confidence": {question_id: {"probs": {...}, "argmax": "..."}}}
    """

    def __init__(self, model: str, questions: list[dict[str, Any]],
                 enable_thinking: bool | None = None):
        import mlx.core as mx
        from mlx_vlm import load

        try:
            mx.set_cache_limit(1 << 30)  # Metal断片化によるGPU Hang対策(Mac既知の問題)
        except Exception:
            pass

        self._mx = mx
        self.questions = questions
        # 思考(reasoning)モデル向けの制御。None=モデル既定に任せる(mlx_vlmは
        # テンプレートが対応していれば既定でenable_thinking=Falseにする)。
        # True/Falseを渡すとチャットテンプレートへ明示的に伝える。
        # 注意: MiniCPM-Vのように enable_thinking を無視して <think> を出すモデルもあり、
        # その場合は思考ぶんを吐き切れるよう max_tokens を上げる必要がある。
        self.enable_thinking = enable_thinking
        print(f"[observe] loading {model} ...", flush=True)
        t0 = time.time()
        self.model, self.processor = load(model)
        self.config = self.model.config
        print(f"[observe] loaded in {time.time()-t0:.1f}s", flush=True)

        tok = self.processor.tokenizer
        self._cand_ids = {}
        for c in questions:
            values = [_as_yaml_safe_str(v) for v in c.get("values", ["yes", "no"])]
            ids = {}
            for v in values:
                enc = tok.encode(v, add_special_tokens=False)
                ids[v] = enc[0] if len(enc) == 1 else None  # 複数トークンに割れる語は計測不能扱い
            self._cand_ids[c["id"]] = ids

    def ask(self, image_path: str, t: float, domain_hint: str, max_tokens: int = 200,
            prefill: str = "") -> dict:
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        mx = self._mx
        prompt = build_prompt(self.questions, domain_hint, t)
        tmpl_kwargs = {}
        if self.enable_thinking is not None:
            tmpl_kwargs["enable_thinking"] = self.enable_thinking
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1, **tmpl_kwargs)
        # アシスタント応答をprefill(例: "{")で始めさせる。Molmoのように
        # プロンプト末尾が"}"で終わると「JSONは完成済み」と誤解して最初のトークンで
        # EOSを出し、空応答になるモデルがある。prefillで生成の口火を切らせる。
        if prefill:
            formatted = formatted + prefill
        try:
            mx.reset_peak_memory()
        except Exception:
            pass

        tok = self.processor.tokenizer
        full_text = prefill  # prefillぶんも含めて有効なJSONとしてパースできるようにする
        confidence: dict[str, dict] = {}
        pending_question = None  # 直前が `"question_id":"` で終わっていれば、次トークンがその値

        for resp in stream_generate(self.model, self.processor, formatted, image=[image_path],
                                     max_tokens=max_tokens, verbose=False):
            if pending_question and resp.logprobs is not None:
                ids = self._cand_ids.get(pending_question, {})
                raw = {}
                for v, tid in ids.items():
                    if tid is None:
                        continue
                    try:
                        raw[v] = math.exp(float(resp.logprobs[tid]))
                    except Exception:
                        pass
                if raw:
                    total = sum(raw.values())
                    probs = {v: round(p / total, 4) for v, p in raw.items()}
                    confidence[pending_question] = {"probs": probs, "argmax": max(probs, key=probs.get)}
                pending_question = None

            tok_id = resp.token
            full_text += tok.decode([tok_id]) if tok_id is not None else ""
            for c in self.questions:
                # モデルがコンパクトJSON("knob":")と整形JSON("knob": ")のどちらを
                # 出力するかは推論ごとに揺れる(このセッションで何度も確認済み)ので、
                # コロン後の空白の有無を問わず検出する。
                if re.search(rf'"{re.escape(c["id"])}"\s*:\s*"$', full_text):
                    pending_question = c["id"]
                    break

        mem = {}
        try:
            mem = {"active_mb": round(mx.get_active_memory() / 1e6, 1),
                   "peak_mb": round(mx.get_peak_memory() / 1e6, 1)}
        except Exception:
            pass
        mx.clear_cache()
        return {"raw": full_text, "confidence": confidence, "mem": mem}


class TransformersObserver:
    """公式transformers実装で観察する代替バックエンド(要torch。--backend transformers)。

    mlx-community変換 + mlx-vlm の経路で視覚入力が壊れるモデル(実測: SmolVLM2)を、
    公式実装のまま同じプロンプト・同じanswer_log形式で観察するために使う。
    プロンプト生成(build_prompt)・prefill・信頼度の計測方法(候補語のlogitを正規化)は
    Observerと同一なので、結果はmlx経路と直接比較できる。
    """

    def __init__(self, model: str, questions: list[dict[str, Any]],
                 enable_thinking: bool | None = None):  # enable_thinkingは互換のため受けて無視
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.questions = questions
        print(f"[observe] loading {model} (transformers) ...", flush=True)
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "mps" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=self.dtype)
        self.model.to(self.device).eval()
        print(f"[observe] loaded in {time.time()-t0:.1f}s (device={self.device})", flush=True)

        tok = self.processor.tokenizer
        self._cand_ids = {}
        for c in questions:
            values = [_as_yaml_safe_str(v) for v in c.get("values", ["yes", "no"])]
            ids = {}
            for v in values:
                enc = tok.encode(v, add_special_tokens=False)
                ids[v] = enc[0] if len(enc) == 1 else None
            self._cand_ids[c["id"]] = ids

    def ask(self, image_path: str, t: float, domain_hint: str, max_tokens: int = 200,
            prefill: str = "") -> dict:
        from PIL import Image

        torch = self._torch
        prompt = build_prompt(self.questions, domain_hint, t)
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                                                  tokenize=False)
        if prefill:
            text = text + prefill
        inputs = self.processor(text=text, images=[Image.open(image_path).convert("RGB")],
                                return_tensors="pt").to(self.device, self.dtype)
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens,
                                      output_scores=True, return_dict_in_generate=True)

        tok = self.processor.tokenizer
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        full_text = prefill
        confidence: dict[str, dict] = {}
        pending_question = None  # Observer.ask と同じ状態機械(直前が `"id":"` なら次トークンが値)

        for i, tok_id in enumerate(gen_ids):
            if pending_question:
                ids = self._cand_ids.get(pending_question, {})
                cand = {v: tid for v, tid in ids.items() if tid is not None}
                if cand:
                    logits = out.scores[i][0]
                    vals = torch.tensor([float(logits[tid]) for tid in cand.values()])
                    probs_t = torch.softmax(vals, dim=0)
                    probs = {v: round(float(p), 4) for v, p in zip(cand, probs_t)}
                    confidence[pending_question] = {"probs": probs,
                                                    "argmax": max(probs, key=probs.get)}
                pending_question = None

            full_text += tok.decode([tok_id])
            for c in self.questions:
                if re.search(rf'"{re.escape(c["id"])}"\s*:\s*"$', full_text):
                    pending_question = c["id"]
                    break

        return {"raw": full_text, "confidence": confidence, "mem": {}}


def confidence_to_answers(confidence: dict[str, dict]) -> dict[str, str]:
    """judge が期待する {question_id: value} 形式に変換する(argmaxだけを取り出す)。"""
    return {question_id: c["argmax"] for question_id, c in confidence.items()}
