import pytest

from src.data import assign_clusters, validate_split_integrity


def test_homology_clusters_never_cross_splits():
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assigned = assign_clusters(rows, {"a": "cluster1", "b": "cluster1", "c": "cluster2"})
    validate_split_integrity(assigned)
    assert assigned[0]["split"] == assigned[1]["split"]


def test_split_integrity_rejects_crossing_cluster():
    rows = [
        {"id": "a", "cluster_id": "x", "split": "train"},
        {"id": "b", "cluster_id": "x", "split": "test"},
    ]
    with pytest.raises(AssertionError):
        validate_split_integrity(rows)

