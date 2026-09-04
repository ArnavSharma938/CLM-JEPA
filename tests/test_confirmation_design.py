import json
from pathlib import Path

from confirmation_design import chemical_pair_id, stable_selection_key, validate_panel


def test_chemical_pair_id_is_directional_and_deterministic():
    first = chemical_pair_id("A.B", "C")
    assert first == chemical_pair_id("A.B", "C")
    assert first != chemical_pair_id("C", "A.B")


def test_selection_key_depends_on_locked_salt():
    pair = chemical_pair_id("A", "B")
    assert stable_selection_key(pair) == stable_selection_key(pair)
    assert stable_selection_key(pair, "different") != stable_selection_key(pair)


def test_repository_confirmation_panel_validates_when_present():
    root = Path(__file__).resolve().parents[1] / "data/clm_jepa_uspto_mit_stp_confirmation"
    panel = root / "untouched_1280.jsonl"
    if panel.exists():
        checks = validate_panel(
            panel, root / "untouched_1280.metadata.json", root / "exclusion_ledger.json"
        )
        assert checks["pair_overlap"] == 0
    amended = root / "untouched_640.jsonl"
    if amended.exists():
        checks = validate_panel(
            amended, root / "untouched_640.metadata.json",
            root / "exclusion_ledger.json", expected_reactions=640,
        )
        assert checks["pair_overlap"] == 0
        original_rows = [json.loads(line) for line in panel.read_text(encoding="utf-8").splitlines()]
        amended_rows = [json.loads(line) for line in amended.read_text(encoding="utf-8").splitlines()]
        assert amended_rows == original_rows[:640]
