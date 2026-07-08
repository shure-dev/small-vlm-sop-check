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
    """SOPの questions: 定義から、1フレーム分の観察プロンプトを自動生成する。"""
    schema_parts = []
    for c in questions:
        values = [_as_yaml_safe_str(v) for v in c.get("values", ["yes", "no"])]
        ask = c["ask"]
        schema_parts.append(f'"{c["id"]}":"{ask} {"/".join(values)}"')
    schema = "{" + ",".join(schema_parts) + "}"
    return (
        f"{domain_hint}（時刻 t={t}s）。"
        "見えている事実だけを、次のJSONで簡潔に答えてください"
        "（憶測禁止・値は指定された選択肢のみ）:\n" + schema
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

    def ask(self, image_path: str, t: float, domain_hint: str, max_tokens: int = 200) -> dict:
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        mx = self._mx
        prompt = build_prompt(self.questions, domain_hint, t)
        tmpl_kwargs = {}
        if self.enable_thinking is not None:
            tmpl_kwargs["enable_thinking"] = self.enable_thinking
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1, **tmpl_kwargs)
        try:
            mx.reset_peak_memory()
        except Exception:
            pass

        tok = self.processor.tokenizer
        full_text = ""
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


def confidence_to_answers(confidence: dict[str, dict]) -> dict[str, str]:
    """judge が期待する {question_id: value} 形式に変換する(argmaxだけを取り出す)。"""
    return {question_id: c["argmax"] for question_id, c in confidence.items()}
