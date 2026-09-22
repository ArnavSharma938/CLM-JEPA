from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Deterministic length-bucketed batches bounded by padded tokens."""

    def __init__(
        self,
        lengths: Sequence[int],
        token_budget: int,
        seed: int,
        *,
        shuffle: bool = True,
        bucket_size: int = 512,
    ) -> None:
        if not lengths or min(lengths) <= 0:
            raise ValueError("all sequence lengths must be positive")
        if token_budget < max(lengths) + 2:
            raise ValueError("token budget cannot fit the longest protein")
        self.lengths = tuple(int(value) + 2 for value in lengths)
        self.token_budget = token_budget
        self.seed = seed
        self.shuffle = shuffle
        self.bucket_size = bucket_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        ordered: list[int] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: (self.lengths[index], index))
            if self.shuffle and (start // self.bucket_size) % 2:
                bucket.reverse()
            ordered.extend(bucket)
        batches: list[list[int]] = []
        current: list[int] = []
        maximum = 0
        for index in ordered:
            candidate_max = max(maximum, self.lengths[index])
            if current and candidate_max * (len(current) + 1) > self.token_budget:
                batches.append(current)
                current, maximum = [], 0
            current.append(index)
            maximum = max(maximum, self.lengths[index])
        if current:
            batches.append(current)
        if self.shuffle:
            random.Random(self.seed ^ self.epoch ^ 0x5A17).shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


def ordered_batch_hash(batches: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update("\0".join(batch).encode())
        digest.update(b"\n")
    return digest.hexdigest()
