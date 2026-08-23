import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_projected_mse_sigreg import add_vectors, space_metrics  # noqa: E402


def test_projected_audit_reports_both_retrieval_geometries():
    values = torch.eye(4)
    metrics = space_metrics(values, values.clone())
    assert metrics["mse_alignment"] == 0.0
    assert metrics["euclidean_retrieval_top1"] == 1.0
    assert metrics["cosine_retrieval_top1"] == 1.0
    assert metrics["pair_centers"]["effective_rank"] > 2.9


def test_projected_audit_full_vector_uses_frozen_component_weights():
    mse = {"layer": torch.tensor([1.0, 2.0])}
    sigreg = {"layer": torch.tensor([3.0, 4.0])}
    result = add_vectors((mse, 2.0), (sigreg, 0.5))
    torch.testing.assert_close(result["layer"], torch.tensor([3.5, 6.0]))
