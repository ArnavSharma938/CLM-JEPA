import numpy as np
import torch

from frozen_geometry import Example, TokenInfo, one_minus_cosine
from stp import PaperSemanticTubePrediction, SemanticTubePrediction
from stp_representation_analysis import (
    checkpoint_specs,
    fixed_spans,
    objective_values,
    semantic_path,
)


def _example():
    segments = ["marker", "source", "source", "marker", "marker", "target", "target", "marker"]
    return Example(
        panel_index=0, reaction_identity="x", source="CC", target="CO", text="",
        input_ids=list(range(8)),
        tokens=[TokenInfo(i, i, i + 1, str(i), segment, i, "atom", 0, 0) for i, segment in enumerate(segments)],
        reaction_center_metadata={},
    )


def test_checkpoint_census_is_locked_and_complete():
    specs = checkpoint_specs()
    assert len(specs) == 22
    assert sum(spec.formulation == "native" for spec in specs) == 5
    assert {(spec.rank, spec.formulation, spec.stp_lambda) for spec in specs} >= {
        (8, "released", .005), (8, "paper", .12), (128, "paper", .02),
    }


def test_semantic_path_matches_paper_reference_mapping():
    example = _example()
    hidden = torch.randn(8, 7)
    path = semantic_path(hidden, example)
    user = torch.tensor((0, 2))
    assistant = torch.tensor((4, 6))
    expected = torch.stack([
        PaperSemanticTubePrediction.semantic_path_embedding(hidden, user, assistant, offset)
        for offset in range(5)
    ])
    torch.testing.assert_close(path, expected)


def test_fixed_objectives_match_released_and_paper_definitions():
    example = _example()
    hidden = torch.randn(8, 11)
    path = semantic_path(hidden, example)
    spans = {
        "released": np.asarray(((0, 2), (1, 4)), dtype=np.int16),
        "paper": np.asarray(((0, 1, 3), (1, 2, 4)), dtype=np.int16),
    }
    released, paper = objective_values(path, spans)
    user = torch.tensor((0, 2))
    assistant = torch.tensor((4, 6))
    expected_released = []
    for start, end in spans["released"]:
        before, patch, after = SemanticTubePrediction.get_embeddings(
            hidden, user, assistant, int(start), int(end),
        )
        expected_released.append(one_minus_cosine((before + after)[None], patch[None])[0])
    expected_paper = []
    for start, interior, end in spans["paper"]:
        expected_paper.append(one_minus_cosine(
            (path[interior] - path[start])[None],
            (path[end] - path[interior])[None],
        )[0])
    torch.testing.assert_close(released, torch.stack(expected_released), atol=1e-6, rtol=0)
    torch.testing.assert_close(paper, torch.stack(expected_paper), atol=1e-6, rtol=0)


def test_paper_diagnostic_spans_reuse_released_outer_constraints():
    example = _example()
    spans = fixed_spans(example, count=1000, seed=19)
    length = int(spans["full_length"])
    assert all(start < interior < end for start, interior, end in spans["paper"])
    assert all(not (start == 0 and end == length) for start, _, end in spans["paper"])
