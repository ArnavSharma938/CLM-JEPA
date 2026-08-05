from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clm_jepa.scoring import load_records, metrics, prediction_records, save_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        targets = [row["tgt"] for row in csv.DictReader(handle)]
    records = prediction_records(result["predictions"], targets)
    save_records(args.records, records)
    offline = metrics(load_records(args.records))
    expected = {key: result[key] for key in ("valid_products", "exact_products")}
    if {key: offline[key] for key in expected} != expected:
        raise AssertionError({"offline": offline, "reported": expected})
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(offline, indent=2) + "\n")
    print(json.dumps(offline))


if __name__ == "__main__":
    main()
