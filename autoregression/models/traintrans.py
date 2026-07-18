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
from autoregression.data.datasplit import (
    cursor_prefix_inputs,
    cursor_prefix_next_ids,
    incorrect_corruption_tensors,
    incorrect_prefix_inputs,
    incorrect_prefix_next_ids,
    training_tensors,
    validation_tensors,
)
from autoregression.data.dataprep import ID_TO_TOKEN, collate

import torch


@torch.no_grad()
def next_token_topk_accuracy(agent, prefix_inputs, next_token_ids, k=5, batch_size=64):
    agent.model.eval()
    device = agent.device
    correct = 0
    total = 0
    for i in range(0, len(prefix_inputs), batch_size):
        batch = prefix_inputs[i : i + batch_size]
        targets = next_token_ids[i : i + batch_size]
        if not batch:
            continue
        lengths = [t.numel() for t in batch]
        max_len = max(lengths)
        input_ids = torch.full((len(batch), max_len), agent.pad_id, dtype=torch.long)
        for r, t in enumerate(batch):
            input_ids[r, : t.numel()] = t
        input_ids = input_ids.to(device)

        logits = agent.model(input_ids)
        last_logits = logits[torch.arange(len(batch), device=device), torch.tensor([l - 1 for l in lengths], device=device)]
        topk = torch.topk(last_logits, k=min(k, last_logits.size(-1)), dim=-1).indices.cpu().tolist()
        for pred_ids, gold in zip(topk, targets):
            correct += int(gold in pred_ids)
        total += len(targets)
    return correct / max(1, total)

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

prefix_top5 = next_token_topk_accuracy(transformer, incorrect_prefix_inputs, incorrect_prefix_next_ids, k=5)
print(f"incorrect-prefix next-token top5 acc: {prefix_top5:.3f}")

cursor_top5 = next_token_topk_accuracy(transformer, cursor_prefix_inputs, cursor_prefix_next_ids, k=5)
print(f"cursor-insertion next-token top5 acc: {cursor_top5:.3f}")

corrupt_ppl = transformer.evaluate(incorrect_corruption_tensors, collate_fn=collate)
print(f"incorrect-corruption ppl: {corrupt_ppl:.3f}")

print(transformer.sample(id_to_token=ID_TO_TOKEN, max_len=40))
transformer.save("transformer.pth")
