from __future__ import annotations

from collections.abc import Sequence

import torch

from .masking import make_full_span_example


class GenerationCollator:
    def __init__(self, special_tokens: dict[str, int]) -> None:
        self.bos_id = int(special_tokens["bos"])
        self.eos_id = int(special_tokens["eos"])
        self.mask_id = int(special_tokens["mask"])
        self.pad_id = int(special_tokens["pad"])

    def __call__(self, examples: list[dict]) -> dict:
        encoded = [make_full_span_example(
            item["receptor_tokens"], item["peptide_tokens"],
            bos_token_id=self.bos_id,
            eos_token_id=self.eos_id,
            mask_token_id=self.mask_id,
        ) for item in examples]
        maximum = max(len(item.input_ids) for item in encoded)
        input_ids = torch.full((len(encoded), maximum), self.pad_id, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, item in enumerate(encoded):
            length = len(item.input_ids)
            input_ids[row_index, :length] = torch.tensor(item.input_ids, dtype=torch.long)
            labels[row_index, :length] = torch.tensor(item.labels, dtype=torch.long)
            attention_mask[row_index, :length] = 1
        supervised_counts = tuple(item.supervised_tokens for item in encoded)
        real_input_tokens = sum(len(item.input_ids) for item in encoded)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "rows": [item["row"] for item in examples],
            "supervised_counts": supervised_counts,
            "supervised_tokens": sum(supervised_counts),
            "padded_input_tokens": input_ids.numel(),
            "real_input_tokens": real_input_tokens,
        }


class GenerationDataset(torch.utils.data.Dataset):
    def __init__(self, cache: dict, manifest_path) -> None:
        import json

        self.cache = cache
        self.rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
        self.items = []
        for row in self.rows:
            cached = cache["pairs"].get(row["pair_id"])
            if cached is None:
                raise RuntimeError(f"pair absent from immutable token cache: {row['pair_id']}")
            self.items.append({
                "row": row,
                "receptor_tokens": cache["receptor_tokens"][row["receptor_id"]],
                "peptide_tokens": cache["peptide_tokens"][cached["peptide_id"]],
            })
        self.lengths = [len(item["receptor_tokens"]) + len(item["peptide_tokens"]) + 2 for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def length_bucket_batches(
    lengths: Sequence[int], token_budget: int, seed: int, epoch: int = 0, shuffle: bool = True,
    bucket_size: int = 512,
) -> list[list[int]]:
    if not lengths or min(lengths) <= 0:
        raise ValueError("all sequence lengths must be positive")
    if token_budget < max(lengths):
        raise ValueError("token budget cannot fit the longest receptor-peptide input")
    import random

    indices = list(range(len(lengths)))
    if shuffle:
        random.Random(seed + epoch).shuffle(indices)
    ordered = []
    for start in range(0, len(indices), bucket_size):
        bucket = indices[start:start + bucket_size]
        bucket.sort(key=lambda index: (lengths[index], index))
        if shuffle and (start // bucket_size) % 2:
            bucket.reverse()
        ordered.extend(bucket)
    batches: list[list[int]] = []
    current: list[int] = []
    maximum = 0
    for index in ordered:
        candidate = max(maximum, int(lengths[index]))
        if current and candidate * (len(current) + 1) > token_budget:
            batches.append(current)
            current, maximum = [], 0
        current.append(index)
        maximum = max(maximum, int(lengths[index]))
    if current:
        batches.append(current)
    if shuffle:
        random.Random(seed ^ epoch ^ 0x6719).shuffle(batches)
    return batches
