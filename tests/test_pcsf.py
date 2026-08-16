import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa import (  # noqa: E402
    PairCenterSpreadFloor,
    pair_center_standard_deviation,
    pair_center_variance,
    pair_centers,
    pcsf_loss,
)
from train import file_sha256, load_pcsf_reference_cache, reaction_row_fingerprint  # noqa: E402


def test_pair_centers_are_exact_matched_barycenters():
    source = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    target = torch.tensor([[3.0, 1.0], [9.0, 3.0]])
    expected = torch.tensor([[2.0, 2.0], [7.0, 5.0]])
    torch.testing.assert_close(pair_centers(source, target), expected)


def test_pair_center_spread_matches_definition():
    centers = torch.tensor([[1.0, 2.0], [3.0, 4.0], [8.0, 1.0]])
    centered = centers - centers.mean(dim=0)
    expected = centered.square().sum() / ((len(centers) - 1) * centers.size(1))
    torch.testing.assert_close(pair_center_variance(centers), expected)


def test_joint_variance_decomposition_with_consistent_normalization():
    torch.manual_seed(533)
    source = torch.randn(11, 17)
    target = torch.randn(11, 17)
    center = pair_centers(source, target)
    joint = torch.cat((source, target), dim=0)
    joint_population_variance = (
        joint - joint.mean(dim=0, keepdim=True)
    ).square().mean()
    center_population_variance = pair_center_variance(center, unbiased=False)
    mse = (source - target).square().mean()
    torch.testing.assert_close(
        joint_population_variance,
        center_population_variance + 0.25 * mse,
        rtol=2e-6,
        atol=2e-7,
    )

    # The prompt's B-1 PCSF convention has the corresponding finite-B factor.
    paired_unbiased_joint = (
        0.5
        * (
            (source - center.mean(dim=0)).square().sum()
            + (target - center.mean(dim=0)).square().sum()
        )
        / ((len(source) - 1) * source.size(1))
    )
    torch.testing.assert_close(
        paired_unbiased_joint,
        pair_center_variance(center)
        + len(source) / (4 * (len(source) - 1)) * mse,
        rtol=2e-6,
        atol=2e-7,
    )


def test_pcsf_is_zero_above_floor_and_positive_below_floor():
    torch.manual_seed(7)
    reference = torch.randn(8, 5)
    high = reference * 1.1
    low = reference * 0.2
    high_loss, _, _ = pcsf_loss(high, high, reference, rho=0.75)
    low_loss, _, _ = pcsf_loss(low, low, reference, rho=0.75)
    assert high_loss.item() == 0.0
    assert low_loss.item() > 0.0


def test_pcsf_has_nonzero_restorative_gradient_under_contraction():
    torch.manual_seed(11)
    reference = torch.randn(9, 6)
    source = (reference * 0.1).requires_grad_(True)
    target = (reference * 0.1).requires_grad_(True)
    loss, sigma_before, _ = pcsf_loss(source, target, reference, rho=0.8)
    loss.backward()
    assert source.grad is not None and source.grad.norm() > 0
    assert target.grad is not None and target.grad.norm() > 0
    with torch.no_grad():
        step_source = source - 0.01 * source.grad
        step_target = target - 0.01 * target.grad
        sigma_after = pair_center_standard_deviation(
            pair_centers(step_source, step_target)
        )
    assert sigma_after > sigma_before


def test_pcsf_is_translation_invariant_and_scales_predictably():
    torch.manual_seed(13)
    source = torch.randn(7, 4)
    target = torch.randn(7, 4)
    reference = torch.randn(7, 4)
    shift = torch.randn(1, 4) * 10
    original = pcsf_loss(source, target, reference, rho=0.9)
    translated = pcsf_loss(
        source + shift, target + shift, reference + shift, rho=0.9,
    )
    torch.testing.assert_close(original[0], translated[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(original[1], translated[1], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(original[2], translated[2], rtol=2e-5, atol=2e-6)

    scale = 2.5
    scaled = pcsf_loss(
        source * scale, target * scale, reference * scale,
        rho=0.9, epsilon=1e-12,
    )
    base = pcsf_loss(source, target, reference, rho=0.9, epsilon=1e-12)
    torch.testing.assert_close(scaled[1], scale * base[1], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(scaled[2], scale * base[2], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(scaled[0], scale**2 * base[0], rtol=2e-5, atol=2e-6)


def test_streamed_pcsf_matches_materialized_value_and_gradients():
    torch.manual_seed(19)
    source_direct = (torch.randn(16, 9) * 0.15).requires_grad_(True)
    target_direct = (torch.randn(16, 9) * 0.15).requires_grad_(True)
    reference = torch.randn(16, 9)
    source_streamed = source_direct.detach().clone().requires_grad_(True)
    target_streamed = target_direct.detach().clone().requires_grad_(True)
    regularizer = PairCenterSpreadFloor(rho=0.8)

    direct_loss, _, _ = regularizer(source_direct, target_direct, reference)
    direct_loss.backward()

    accumulator = regularizer.start_streaming(expected_samples=16, dimensions=9)
    offset = 0
    for width in (3, 5, 8):
        end = offset + width
        accumulator.update(
            source_streamed[offset:end],
            target_streamed[offset:end],
            reference[offset:end],
        )
        offset = end
    prepared = accumulator.finalize()
    surrogate = sum(
        prepared.surrogate(source_chunk, target_chunk)
        for source_chunk, target_chunk in zip(
            source_streamed.split((3, 5, 8)),
            target_streamed.split((3, 5, 8)),
        )
    )
    surrogate.backward()

    torch.testing.assert_close(prepared.loss, direct_loss.detach(), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(source_streamed.grad, source_direct.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(target_streamed.grad, target_direct.grad, rtol=2e-5, atol=2e-6)


def test_reference_lookup_preserves_exact_logical_batch_identities(tmp_path):
    manifest = tmp_path / "train.csv"
    manifest.write_text("source,target\nA,B\nC,D\n", encoding="utf-8")
    rows = [
        {"source": "A", "target": "B", "src": "A", "tgt": "B"},
        {"source": "C", "target": "D", "src": "C", "tgt": "D"},
    ]
    cache_path = tmp_path / "reference.pt"
    torch.save({
        "schema_version": 1,
        "train_manifest_sha256": file_sha256(manifest),
        "row_fingerprints": [reaction_row_fingerprint(row) for row in rows],
        "pair_centers": torch.arange(8, dtype=torch.float32).reshape(2, 4),
    }, cache_path)
    loaded, _ = load_pcsf_reference_cache(cache_path, rows, manifest)
    assert [row["pcsf_reference_index"] for row in rows] == ["0", "1"]
    torch.testing.assert_close(
        loaded, torch.arange(8, dtype=torch.float32).reshape(2, 4),
    )

    cache = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    batch_indices = torch.tensor([7, 1, 9, 3])
    expected = torch.stack((cache[7], cache[1], cache[9], cache[3]))
    torch.testing.assert_close(cache.index_select(0, batch_indices), expected)
