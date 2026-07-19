import sys
from pathlib import Path

import onnx
import torch


repo_root = Path(__file__).resolve().parents[1]  # repo root (parent of autoregression/)
sys.path.insert(0, str(repo_root))

from autoregression.models.transformer import TransformerAgent


def find_weights(default_path):
    if default_path.exists():
        return default_path

    for filename in ("transformer.pth", "trans.pth"):
        candidates = list(repo_root.rglob(filename))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"Could not find transformer weights at {default_path}. "
        "Set weights_path to your .pth location, or place transformer.pth in repo root."
    )


def main():
    weights_path = repo_root / "autoregression" / "trans.pth"
    onnx_path = repo_root / "transformer.onnx"
    single_file_onnx_path = repo_root / "transformer_single.onnx"
    max_len = 64
    inline_weights = True

    agent = TransformerAgent(
        emb_dim=128,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        max_len=max_len,
        lr=3e-4,
        device="cpu",
    )

    weights_path = find_weights(weights_path)
    agent.load(str(weights_path))
    agent.model.eval()

    dummy = torch.zeros((1, max_len), dtype=torch.long)

    export_kwargs = dict(
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    try:
        torch.onnx.export(
            agent.model,
            dummy,
            str(onnx_path),
            **export_kwargs,
            use_external_data_format=not inline_weights,
        )
    except TypeError:
        torch.onnx.export(agent.model, dummy, str(onnx_path), **export_kwargs)

    print("wrote", str(onnx_path))

    if inline_weights:
        model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save_model(model, str(single_file_onnx_path), save_as_external_data=False)
        print("wrote", str(single_file_onnx_path))


if __name__ == "__main__":
    main()
