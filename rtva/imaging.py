"""processor 画像前処理の実効制御。

回避5(transformers の Qwen2VLImageProcessor): `.max_pixels` 属性は無視される
(実測: 720x540 入力で設定値によらず grid (1,34,44)=374トークン/フレーム)。
実効ノブは `size` dict で、`longest_edge` が事実上の max_pixels として働く
(smart_resize が総ピクセル数を longest_edge 以下へ丸める。
実測: longest_edge=112896 -> grid (1,18,24)=108トークン/フレーム)。

この見落としが v3 学習の Metal OOM の真因だった: 10フレーム×374tok で
系列約4,500(動作実績 v1 の2倍)になっていた。学習/評価は必ずこの関数を使うこと。
"""


def set_max_image_pixels(processor, max_px: int) -> None:
    """学習・評価で同一値を使う(トークン数=入力条件を一致させるため)。"""
    ip = processor.image_processor
    ip.max_pixels = max_px  # 属性を参照する実装にも一応効かせる(現行では無視される)
    ip.size = {"shortest_edge": min(65536, max_px), "longest_edge": max_px}
