import json
import sys

import pytest

from slice_official_panel import main, read_jsonl


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_slice_preserves_manifest_order_and_aligns_predictions(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output_manifest = tmp_path / "prefix.jsonl"
    output_predictions = tmp_path / "prefix_predictions.jsonl"
    metadata = tmp_path / "metadata.json"
    rows = [
        {"reaction_identity": f"r{index}", "panel_index": index}
        for index in range(4)
    ]
    write_jsonl(manifest, rows)
    write_jsonl(predictions, [
        {"reaction_identity": "r2", "value": 2},
        {"reaction_identity": "r0", "value": 0},
        {"reaction_identity": "r3", "value": 3},
        {"reaction_identity": "r1", "value": 1},
    ])
    monkeypatch.setattr(sys, "argv", [
        "slice_official_panel.py",
        "--manifest", str(manifest),
        "--output-manifest", str(output_manifest),
        "--limit", "3",
        "--predictions", str(predictions), str(output_predictions),
        "--metadata", str(metadata),
    ])
    main()
    assert [row["reaction_identity"] for row in read_jsonl(output_manifest)] == [
        "r0", "r1", "r2",
    ]
    assert [row["value"] for row in read_jsonl(output_predictions)] == [0, 1, 2]
    assert json.loads(metadata.read_text())["limit"] == 3


def test_slice_rejects_missing_prediction_identity(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(manifest, [
        {"reaction_identity": "r0", "panel_index": 0},
        {"reaction_identity": "r1", "panel_index": 1},
    ])
    write_jsonl(predictions, [{"reaction_identity": "r0"}])
    monkeypatch.setattr(sys, "argv", [
        "slice_official_panel.py",
        "--manifest", str(manifest),
        "--output-manifest", str(tmp_path / "prefix.jsonl"),
        "--limit", "2",
        "--predictions", str(predictions), str(tmp_path / "pred_prefix.jsonl"),
        "--metadata", str(tmp_path / "metadata.json"),
    ])
    with pytest.raises(ValueError, match="missing 1 prefix identities"):
        main()
