"""
Entry-point for training the Transformer (GPT-style) autoregressive model.

Recommended usage from repo root:
  python -m autoregression.models.traintransformer

This file also supports being run directly (e.g. by VS Code) via a small
sys.path bootstrap below.
"""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from autoregression.models.transformer import TransformerAgent
from autoregression.data.datasplit import training_tensors, validation_tensors
from autoregression.data.dataprep import ID_TO_TOKEN, collate

transformer = TransformerAgent(
        emb_dim=128,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        max_len=64,
        lr=3e-4)

transformer.fit(training_tensors, validation_tensors, collate_fn=collate, epochs=20)

val_ppl = transformer.evaluate(validation_tensors, collate_fn=collate)
print(f"final val ppl: {val_ppl:.3f}")

print(transformer.sample(id_to_token=ID_TO_TOKEN, max_len=40))
transformer.save("transformer.pth")