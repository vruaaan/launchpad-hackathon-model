import sys
from pathlib import Path
import torch
import onnx

repo_root = Path(__file__).resolve().parents[1]  # repo root (parent of autoregression/)
sys.path.insert(0, str(repo_root))
from autoregression.models.lstm import RNNAgent

def main():
    # Adjust if you saved elsewhere. If missing, we'll attempt to locate it.
    weights_path = repo_root / "lstm.pth"
    onnx_path = repo_root / "lstm.onnx"
    single_file_onnx_path = repo_root / "lstm_single.onnx"
    max_len = 64                # pick a max you'll use for inference
    inline_weights = True       # writes a 1-file ONNX (no .onnx.data)

    agent = RNNAgent(
        emb_dim=128, hidden_dim=256, num_layers=2,
        dropout=0.2, lr=1e-3, device="cpu"
    )
    if not weights_path.exists():
        candidates = list(repo_root.rglob("lstm.pth"))
        if candidates:
            weights_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Could not find lstm.pth at {weights_path}. "
                "Set weights_path to your .pth location (or place lstm.pth in repo root)."
            )

    agent.load(str(weights_path))
    agent.model.eval()

    dummy = torch.zeros((1, max_len), dtype=torch.long)

    # Exports forward(input_ids) -> (logits, hidden). Hidden is awkward in ONNX;
    # simplest is to export logits only by wrapping.
    class Wrapper(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core
        def forward(self, input_ids):
            logits, _hidden = self.core(input_ids)
            return logits

    wrapped = Wrapper(agent.model)

    export_kwargs = dict(
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "time"},
            "logits": {0: "batch", 1: "time"},
        },
        opset_version=17,
    )
    try:
        torch.onnx.export(
            wrapped,
            dummy,
            str(onnx_path),
            **export_kwargs,
            use_external_data_format=not inline_weights,
        )
    except TypeError:
        torch.onnx.export(wrapped, dummy, str(onnx_path), **export_kwargs)
    print("wrote", str(onnx_path))

    if inline_weights:
        m = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save_model(m, str(single_file_onnx_path), save_as_external_data=False)
        print("wrote", str(single_file_onnx_path))

if __name__ == "__main__":
    main()
