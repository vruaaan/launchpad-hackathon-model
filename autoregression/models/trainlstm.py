"""
Entry-point for training the LSTM autoregressive model.

Recommended usage from repo root:
  python -m autoregression.models.trainlstm

This file also supports being run directly (e.g. by VS Code) via a small
sys.path bootstrap below.
"""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    # Running as a script: add repo root so `import autoregression...` works.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from autoregression.models.lstm import RNNAgent
from autoregression.data.datasplit import training_tensors, validation_tensors
from autoregression.data.dataprep import ID_TO_TOKEN, collate

rnn = RNNAgent(
        emb_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
        lr=1e-3)
 
rnn.fit(training_tensors, validation_tensors, collate_fn=collate, epochs=20)
 
val_ppl = rnn.evaluate(validation_tensors, collate_fn=collate)
print(f"final val ppl: {val_ppl:.3f}")
 
print(rnn.sample(id_to_token=ID_TO_TOKEN, max_len=40))
rnn.save("lstm.pth")
