import math
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

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


def _model_logits(agent, input_ids):
    output = agent.model(input_ids)
    return output[0] if isinstance(output, tuple) else output


def _token_name(token_id, id_to_token):
    return id_to_token[int(token_id)]


def _sequence_case_labels(sequence, id_to_token):
    tokens = [_token_name(token_id, id_to_token) for token_id in sequence.tolist()]
    labels = {"all"}
    if "<CURSOR>" in tokens:
        labels.add("has_cursor")
    if "SUBQUERY_START" in tokens:
        labels.add("subquery")
    if "EXISTS" in tokens:
        labels.add("exists")
    if "BETWEEN" in tokens:
        labels.add("between")
    if "ABS_DATE" in tokens:
        labels.add("abs_date")
    if "MATH_OP" in tokens:
        labels.add("math_op")
    if "SELECT_ITEM" in tokens:
        labels.add("select_item")
    if "JOIN" in tokens:
        labels.add("join")
    if "GROUP_BY" in tokens:
        labels.add("group_by")
    if "HAVING" in tokens:
        labels.add("having")
    if "ORDER_BY" in tokens:
        labels.add("order_by")
    if "LIMIT" in tokens:
        labels.add("limit")
    if "AS" in tokens or "ALIAS" in tokens:
        labels.add("alias")
    if "AND" in tokens or "OR" in tokens:
        labels.add("boolean_chain")
    return labels


def _prefix_case_labels(prefix, target_id, id_to_token):
    tokens = [_token_name(token_id, id_to_token) for token_id in prefix.tolist()]
    target = _token_name(target_id, id_to_token)
    labels = _sequence_case_labels(prefix, id_to_token)
    labels.add(f"target:{target}")
    if tokens and tokens[-1] == "<CURSOR>":
        labels.add("cursor_query")
    if len(tokens) <= 4:
        labels.add("short_prefix")
    elif len(tokens) >= 12:
        labels.add("long_prefix")
    return labels


@torch.no_grad()
def _sequence_ppl_by_case(agent, sequences, id_to_token, collate_fn, batch_size=64):
    agent.model.eval()
    device = agent.device
    criterion = torch.nn.CrossEntropyLoss(ignore_index=agent.pad_id, reduction="none")
    totals = defaultdict(lambda: {"loss": 0.0, "tokens": 0, "examples": 0})

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        if not batch:
            continue
        input_ids, target_ids = collate_fn(batch)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        logits = _model_logits(agent, input_ids)
        losses = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        ).reshape(target_ids.shape)
        real_mask = target_ids != agent.pad_id
        per_example_loss = (losses * real_mask).sum(dim=1).cpu().tolist()
        per_example_tokens = real_mask.sum(dim=1).cpu().tolist()

        for sequence, loss_sum, token_count in zip(batch, per_example_loss, per_example_tokens):
            if token_count == 0:
                continue
            for label in _sequence_case_labels(sequence, id_to_token):
                totals[label]["loss"] += float(loss_sum)
                totals[label]["tokens"] += int(token_count)
                totals[label]["examples"] += 1

    results = {}
    for label, values in totals.items():
        results[label] = {
            "ppl": math.exp(values["loss"] / max(1, values["tokens"])),
            "tokens": values["tokens"],
            "examples": values["examples"],
        }
    return results


@torch.no_grad()
def _topk_by_case(agent, prefix_inputs, next_token_ids, id_to_token, k=5, batch_size=64):
    agent.model.eval()
    device = agent.device
    totals = defaultdict(lambda: {"correct": 0, "total": 0, "misses": []})

    for i in range(0, len(prefix_inputs), batch_size):
        batch = prefix_inputs[i : i + batch_size]
        targets = next_token_ids[i : i + batch_size]
        if not batch:
            continue
        lengths = [t.numel() for t in batch]
        padded = pad_sequence(batch, batch_first=True, padding_value=agent.pad_id).to(device)
        logits = _model_logits(agent, padded)
        last_logits = logits[
            torch.arange(len(batch), device=device),
            torch.tensor([length - 1 for length in lengths], device=device),
        ]
        topk = torch.topk(last_logits, k=min(k, last_logits.size(-1)), dim=-1).indices.cpu().tolist()

        for prefix, pred_ids, target_id in zip(batch, topk, targets):
            labels = _prefix_case_labels(prefix, target_id, id_to_token)
            is_correct = int(target_id in pred_ids)
            miss = {
                "target": _token_name(target_id, id_to_token),
                "topk": [_token_name(pred_id, id_to_token) for pred_id in pred_ids],
                "prefix": [_token_name(token_id, id_to_token) for token_id in prefix.tolist()],
            }
            for label in labels:
                totals[label]["correct"] += is_correct
                totals[label]["total"] += 1
                if not is_correct and len(totals[label]["misses"]) < 5:
                    totals[label]["misses"].append(miss)

    results = {}
    for label, values in totals.items():
        results[label] = {
            "accuracy": values["correct"] / max(1, values["total"]),
            "correct": values["correct"],
            "total": values["total"],
            "misses": values["misses"],
        }
    return results


def _worst_accuracy_cases(results, min_total=20, limit=8):
    rows = [
        (label, values)
        for label, values in results.items()
        if values["total"] >= min_total and label != "all"
    ]
    return sorted(rows, key=lambda item: (item[1]["accuracy"], -item[1]["total"]))[:limit]


def _worst_ppl_cases(results, min_examples=10, limit=8):
    rows = [
        (label, values)
        for label, values in results.items()
        if values["examples"] >= min_examples and label != "all"
    ]
    return sorted(rows, key=lambda item: (-item[1]["ppl"], -item[1]["examples"]))[:limit]


def perf(
    agent,
    *,
    id_to_token=None,
    collate_fn=None,
    validation_tensors=None,
    incorrect_corruption_tensors=None,
    incorrect_prefix_inputs=None,
    incorrect_prefix_next_ids=None,
    cursor_prefix_inputs=None,
    cursor_prefix_next_ids=None,
    k=5,
    batch_size=64,
    min_total=20,
    print_report=True,
):
    """
    Runs model evaluation and groups weaker performance by case type.

    Returns a dictionary with:
      - validation perplexity by token/case bucket
      - corruption perplexity by token/case bucket
      - incomplete-prefix top-k accuracy by target/case bucket
      - cursor-insertion top-k accuracy by target/case bucket

    Example:
        from autoregression.eval import perf
        report = perf(rnn)
    """
    if any(
        value is None
        for value in (
            id_to_token,
            collate_fn,
            validation_tensors,
            incorrect_corruption_tensors,
            incorrect_prefix_inputs,
            incorrect_prefix_next_ids,
            cursor_prefix_inputs,
            cursor_prefix_next_ids,
        )
    ):
        from autoregression.data.dataprep import ID_TO_TOKEN, collate
        from autoregression.data.datasplit import (
            cursor_prefix_inputs as default_cursor_prefix_inputs,
            cursor_prefix_next_ids as default_cursor_prefix_next_ids,
            incorrect_corruption_tensors as default_incorrect_corruption_tensors,
            incorrect_prefix_inputs as default_incorrect_prefix_inputs,
            incorrect_prefix_next_ids as default_incorrect_prefix_next_ids,
            validation_tensors as default_validation_tensors,
        )

        id_to_token = id_to_token or ID_TO_TOKEN
        collate_fn = collate_fn or collate
        validation_tensors = validation_tensors or default_validation_tensors
        incorrect_corruption_tensors = incorrect_corruption_tensors or default_incorrect_corruption_tensors
        incorrect_prefix_inputs = incorrect_prefix_inputs or default_incorrect_prefix_inputs
        incorrect_prefix_next_ids = incorrect_prefix_next_ids or default_incorrect_prefix_next_ids
        cursor_prefix_inputs = cursor_prefix_inputs or default_cursor_prefix_inputs
        cursor_prefix_next_ids = cursor_prefix_next_ids or default_cursor_prefix_next_ids

    validation_cases = _sequence_ppl_by_case(
        agent, validation_tensors, id_to_token, collate_fn, batch_size=batch_size
    )
    corruption_cases = _sequence_ppl_by_case(
        agent, incorrect_corruption_tensors, id_to_token, collate_fn, batch_size=batch_size
    )
    prefix_cases = _topk_by_case(
        agent, incorrect_prefix_inputs, incorrect_prefix_next_ids, id_to_token, k=k, batch_size=batch_size
    )
    cursor_cases = _topk_by_case(
        agent, cursor_prefix_inputs, cursor_prefix_next_ids, id_to_token, k=k, batch_size=batch_size
    )

    report = {
        "validation": {
            "overall_ppl": validation_cases.get("all", {}).get("ppl"),
            "by_case": validation_cases,
            "worst_cases": _worst_ppl_cases(validation_cases, min_examples=min_total),
        },
        "corruption": {
            "overall_ppl": corruption_cases.get("all", {}).get("ppl"),
            "by_case": corruption_cases,
            "worst_cases": _worst_ppl_cases(corruption_cases, min_examples=min_total),
        },
        "incorrect_prefix": {
            "overall_topk_accuracy": prefix_cases.get("all", {}).get("accuracy"),
            "by_case": prefix_cases,
            "worst_cases": _worst_accuracy_cases(prefix_cases, min_total=min_total),
        },
        "cursor_insertion": {
            "overall_topk_accuracy": cursor_cases.get("all", {}).get("accuracy"),
            "by_case": cursor_cases,
            "worst_cases": _worst_accuracy_cases(cursor_cases, min_total=min_total),
        },
    }

    if print_report:
        print(f"validation ppl: {report['validation']['overall_ppl']:.3f}")
        print(f"corruption ppl: {report['corruption']['overall_ppl']:.3f}")
        print(f"incorrect-prefix top{k} acc: {report['incorrect_prefix']['overall_topk_accuracy']:.3f}")
        print(f"cursor-insertion top{k} acc: {report['cursor_insertion']['overall_topk_accuracy']:.3f}")

        print("\nWorst validation ppl cases:")
        for label, values in report["validation"]["worst_cases"]:
            print(f"  {label}: ppl={values['ppl']:.3f}, examples={values['examples']}")

        print("\nWorst corruption ppl cases:")
        for label, values in report["corruption"]["worst_cases"]:
            print(f"  {label}: ppl={values['ppl']:.3f}, examples={values['examples']}")

        print("\nWorst incorrect-prefix top-k cases:")
        for label, values in report["incorrect_prefix"]["worst_cases"]:
            print(f"  {label}: acc={values['accuracy']:.3f}, total={values['total']}")

        print("\nWorst cursor-insertion top-k cases:")
        for label, values in report["cursor_insertion"]["worst_cases"]:
            print(f"  {label}: acc={values['accuracy']:.3f}, total={values['total']}")

    return report
