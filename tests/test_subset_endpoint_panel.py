import json

import pytest

from scripts.subset_endpoint_panel import subset_panel


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )


def test_subset_panel_orders_shards_and_ignores_out_of_panel_rows(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    shard_a = tmp_path / "a.jsonl"
    shard_b = tmp_path / "b.jsonl"
    rows = [{"reaction_identity": name} for name in ("a", "b", "c")]
    _write(manifest, rows)
    _write(shard_a, [{"reaction_identity": "c"}, {"reaction_identity": "a"}])
    _write(shard_b, [{"reaction_identity": "b"}])

    panel, predictions, ignored = subset_panel(manifest, [shard_a, shard_b], 2)

    assert [row["reaction_identity"] for row in panel] == ["a", "b"]
    assert [row["reaction_identity"] for row in predictions] == ["a", "b"]
    assert ignored == 1


def test_subset_panel_rejects_missing_identity(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    shard = tmp_path / "shard.jsonl"
    _write(manifest, [{"reaction_identity": "a"}, {"reaction_identity": "b"}])
    _write(shard, [{"reaction_identity": "a"}])

    with pytest.raises(ValueError, match="miss 1"):
        subset_panel(manifest, [shard], 2)
