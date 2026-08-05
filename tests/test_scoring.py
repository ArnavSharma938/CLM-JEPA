import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clm_jepa.scoring import load_records, metrics, prediction_records, save_records


def test_saved_predictions_reproduce_metrics_offline(tmp_path):
    rows = prediction_records(["CCO", "not-smiles", "O=C=O"], ["OCC", "CC", "O=C=O"])
    online = metrics(rows)
    path = tmp_path / "predictions.jsonl"
    save_records(path, rows)
    assert metrics(load_records(path)) == online == {"count": 3, "valid_products": 2, "exact_products": 2}
