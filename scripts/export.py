"""export.py — 学習済み UsageClassifier を配布形式へ書き出す（量子化＝評価の E）。

実証する主張: 「速度」。CPU 即応のため ONNX + 動的量子化(int8) を第一級の配布物とする。
MLX / GGUF はエンコーダ分類器では実験的なため、可能なら変換し、ツールが無ければ
正確な手順を案内する（見かけだけの成功を出さない）。

使い方:
  python scripts/export.py --model artifacts/scribe-usage --out exports [--push REPO_ID]
出力:
  exports/onnx/            … ONNX(FP32)
  exports/onnx-int8/       … ONNX 動的量子化(int8)  ← 評価(E)・本番CPUの既定
  exports/mlx/             … MLX（可能な場合）
  exports/gguf/            … GGUF（可能な場合、案内のみのことあり）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def export_onnx(model_dir: str, out_dir: Path):
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ORTModelForTokenClassification.from_pretrained(model_dir, export=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_dir).save_pretrained(out_dir)
    print(f"ONNX(FP32): {out_dir}")


def quantize_onnx(onnx_dir: Path, out_dir: Path):
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    out_dir.mkdir(parents=True, exist_ok=True)
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)
    print(f"ONNX(int8 動的量子化): {out_dir}  ← 評価(E)/本番CPUの既定")


def export_mlx(model_dir: str, out_dir: Path):
    try:
        import mlx.core  # noqa
    except Exception:
        print("MLX 未導入: `pip install mlx` 後に再実行してください（Apple Silicon 推奨）。")
        return
    try:
        # エンコーダ分類器の MLX 変換は用途特化のため、重みを npz で書き出す最小実装。
        import numpy as np
        import mlx.core as mx
        from transformers import AutoModelForTokenClassification
        out_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        weights = {k: mx.array(v.detach().cpu().numpy()) for k, v in model.state_dict().items()}
        mx.save_safetensors(str(out_dir / "weights.safetensors"), weights)
        print(f"MLX(weights): {out_dir}（推論ラッパは別途 mlx で実装）")
    except Exception as e:
        print(f"MLX 変換をスキップ: {e}")


def export_gguf(model_dir: str, out_dir: Path):
    """GGUF は llama.cpp のデコーダ系 LM が主対象。BERT 分類器は非標準。

    可能なら llama.cpp の変換を案内し、既定は ONNX(int8)/MLX を推奨する。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    guide = (
        "GGUF について:\n"
        "  本モデルはエンコーダ(BERT)ベースの token 分類器で、GGUF(llama.cpp)の主対象である\n"
        "  デコーダ系 LM とは異なります。CPU 配布は ONNX(int8) を第一に推奨します。\n"
        "  BERT の埋め込み用途で GGUF 化する場合は llama.cpp の convert_hf_to_gguf.py\n"
        "  (--outtype q8_0 等) を参照してください。分類ヘッドは GGUF では別途扱いが必要です。\n"
    )
    (out_dir / "README.txt").write_text(guide, encoding="utf-8")
    print(guide.strip())


def push(repo_id: str, paths):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    for p in paths:
        if Path(p).exists():
            api.upload_folder(folder_path=str(p), path_in_repo=Path(p).name, repo_id=repo_id)
    print(f"push 完了: https://huggingface.co/{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="学習済みモデルのディレクトリ")
    ap.add_argument("--out", default=str(ROOT / "exports"))
    ap.add_argument("--push", metavar="REPO_ID")
    ap.add_argument("--skip-mlx", action="store_true")
    ap.add_argument("--skip-gguf", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    onnx_dir = out / "onnx"
    int8_dir = out / "onnx-int8"

    export_onnx(args.model, onnx_dir)
    quantize_onnx(onnx_dir, int8_dir)
    if not args.skip_mlx:
        export_mlx(args.model, out / "mlx")
    if not args.skip_gguf:
        export_gguf(args.model, out / "gguf")

    if args.push:
        push(args.push, [onnx_dir, int8_dir, out / "mlx", out / "gguf"])


if __name__ == "__main__":
    main()
