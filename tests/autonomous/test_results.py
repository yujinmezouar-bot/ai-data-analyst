import pandas as pd

from autonomous.results import Finding, FindingsStore


def test_findings_store_starts_empty():
    store = FindingsStore()
    assert len(store) == 0
    assert store.all() == []


def test_findings_store_records_and_gets_finding():
    store = FindingsStore()
    finding = Finding(
        id="f1",
        step_id="step_1",
        tool_name="statistics",
        datasets=["sales"],
        result={"mean": 10},
        metadata={"objective": "Analyze sales"},
        provenance={"step_id": "step_1"},
    )
    stored = store.record(finding)
    assert stored.id == "f1"
    assert store.get("f1") == finding
    assert len(store) == 1


def test_findings_store_finds_by_step_and_tool():
    store = FindingsStore()
    store.record({
        "id": "f1",
        "step_id": "s1",
        "tool_name": "dataset_info",
        "datasets": ["sales"],
        "result": {"rows": 10},
        "metadata": {"a": 1},
        "provenance": {"source": "sales"},
    })
    store.record({
        "id": "f2",
        "step_id": "s2",
        "tool_name": "statistics",
        "datasets": ["sales"],
        "result": {"mean": 5},
        "metadata": {"a": 2},
        "provenance": {"source": "sales"},
    })

    assert [f.id for f in store.find_by_step("s1")] == ["f1"]
    assert [f.id for f in store.find_by_tool("statistics")] == ["f2"]
    assert len(store.find_by_tool("dataset_info")) == 1


def test_findings_store_clear_removes_all():
    store = FindingsStore()
    store.record({"id": "f1", "step_id": "s1", "tool_name": "dataset_info", "datasets": ["sales"], "result": {}})
    store.clear()
    assert len(store) == 0
    assert store.all() == []


def test_findings_store_preserves_provenance_metadata():
    store = FindingsStore()
    store.record({
        "id": "f1",
        "step_id": "s1",
        "tool_name": "statistics",
        "datasets": ["sales"],
        "result": {"count": 2},
        "metadata": {"column": "revenue"},
        "provenance": {"dataset_names": ["sales"], "plan_id": "plan_1"},
    })

    finding = store.get("f1")
    assert finding is not None
    assert finding.metadata["column"] == "revenue"
    assert finding.provenance["plan_id"] == "plan_1"
