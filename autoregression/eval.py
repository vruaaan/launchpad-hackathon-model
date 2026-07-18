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

        output = agent.model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output
        last_logits = logits[torch.arange(len(batch), device=device), torch.tensor([l - 1 for l in lengths], device=device)]
        topk = torch.topk(last_logits, k=min(k, last_logits.size(-1)), dim=-1).indices.cpu().tolist()
        for pred_ids, gold in zip(topk, targets):
            correct += int(gold in pred_ids)
        total += len(targets)
    return correct / max(1, total)
