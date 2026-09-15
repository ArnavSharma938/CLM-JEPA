from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from src.fitness import official_rita_fitness


class TinyTokenizer:
    table = {"A": 0, "C": 1, "D": 3}

    def encode(self, sequence):
        return [self.table[x] for x in sequence] + [2]


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([
            [3.0, 0.0, -1.0, 1.0], [0.0, 3.0, 1.0, -1.0],
            [-1.0, 0.0, 3.0, 1.0], [1.0, -1.0, 0.0, 3.0],
        ]), requires_grad=False)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.logits[input_ids])


def test_fitness_matches_official_forward_plus_reverse_mean_ce():
    model, tokenizer, sequence = TinyLM(), TinyTokenizer(), "ACD"
    actual = official_rita_fitness(model, tokenizer, sequence, "cpu")
    expected = 0.0
    for direction in (sequence, sequence[::-1]):
        ids = torch.tensor([tokenizer.encode(direction)])
        logits = model(ids[:, :-1]).logits
        expected -= float(F.cross_entropy(logits.reshape(-1, 4), ids[:, 1:].reshape(-1)))
    assert actual == expected

