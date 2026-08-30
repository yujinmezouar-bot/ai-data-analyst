import pandas as pd
import pytest

from autonomous.executor import Executor, ExecutorError
from autonomous.plan import AnalysisPlan, PlanStep
from autonomous.results import FindingsStore
from tools.dataset_info import dataset_info


@pytest.fixture
def customers_orders_datasets():
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["A", "B", "C"],
    })
    orders = pd.DataFrame({
        "order_id": [10, 11, 12, 13],
        "customer_id": [1, 1, 2, 3],
        "amount": [100, 150, 200, 50],
    })
    return {"customers": customers, "orders": orders}


@pytest.fixture
def orders_items_datasets():
    orders = pd.DataFrame({
        "order_id": [10, 11, 12, 13],
        "customer_id": [1, 1, 2, 3],
    })
    items = pd.DataFrame({
        "order_id": [10, 11, 10, 12],
        "item": ["x", "y", "z", "w"],
    })
    return {"orders": orders, "items": items}


def test_discover_relationships_records_finding(customers_orders_datasets):
    store = FindingsStore()
    executor = Executor({"dataset_info": dataset_info}, findings_store=store)

    plan = AnalysisPlan(
        id="p1",
        objective="discover",
        datasets=["customers", "orders"],
        steps=[PlanStep(id="s1", tool_name="discover_relationships", kwargs={}, read_only=True)],
    )

    findings = executor.execute(plan, customers_orders_datasets)
    all_findings = findings.all()
    assert len(all_findings) == 1
    f = all_findings[0]
    assert f.tool_name == "discover_relationships"
    assert "relationships" in f.result or "relationships_found" in f.result


def test_discover_relationships_invalid_dataset_rejected(customers_orders_datasets):
    store = FindingsStore()
    executor = Executor({"dataset_info": dataset_info}, findings_store=store)
    plan = AnalysisPlan(
        id="p2",
        objective="discover",
        datasets=["customers", "orders"],
        steps=[PlanStep(id="s1", tool_name="discover_relationships", kwargs={"dataset_name": "missing"}, read_only=True)],
    )
    with pytest.raises(ExecutorError):
        executor.execute(plan, customers_orders_datasets)


def test_inspect_join_viability_and_record(customers_orders_datasets):
    store = FindingsStore()
    executor = Executor({"dataset_info": dataset_info}, findings_store=store)
    plan = AnalysisPlan(
        id="p3",
        objective="check join",
        datasets=["customers", "orders"],
        steps=[
            PlanStep(
                id="s1",
                tool_name="inspect_join_viability",
                kwargs={"left_dataset": "customers", "right_dataset": "orders", "left_on": "customer_id", "right_on": "customer_id"},
                read_only=True,
            )
        ],
    )
    findings = executor.execute(plan, customers_orders_datasets)
    f = findings.all()[0]
    assert f.tool_name == "inspect_join_viability"
    assert f.result["safe_to_join"] is True


def test_execute_join_rejects_unsafe_join(orders_items_datasets):
    store = FindingsStore()
    executor = Executor({}, findings_store=store)
    plan = AnalysisPlan(
        id="p4",
        objective="join",
        datasets=["orders", "items"],
        steps=[
            PlanStep(
                id="s1",
                tool_name="execute_join",
                kwargs={"left_dataset": "orders", "right_dataset": "items", "left_on": "customer_id", "right_on": "order_id"},
                read_only=False,
            )
        ],
    )
    with pytest.raises(ExecutorError):
        executor.execute(plan, orders_items_datasets)


def test_successful_join_and_derived_availability(customers_orders_datasets):
    store = FindingsStore()
    registered = {}

    def registrar(name, df):
        registered[name] = df

    executor = Executor({"dataset_info": dataset_info}, findings_store=store, derived_dataset_register=registrar)

    plan = AnalysisPlan(
        id="p5",
        objective="join",
        datasets=["customers", "orders"],
        steps=[
            PlanStep(id="s1", tool_name="inspect_join_viability", kwargs={"left_dataset": "customers", "right_dataset": "orders", "left_on": "customer_id", "right_on": "customer_id"}, read_only=True),
            PlanStep(id="s2", tool_name="execute_join", kwargs={"left_dataset": "customers", "right_dataset": "orders", "left_on": "customer_id", "right_on": "customer_id", "how": "inner"}, read_only=False),
            PlanStep(id="s3", tool_name="dataset_info", kwargs={"dataset_name": "derived_join_1"}, read_only=True),
        ],
    )

    findings = executor.execute(plan, customers_orders_datasets)
    # Check that derived dataset was registered
    assert "derived_join_1" in registered
    # Check findings include inspect and execute
    names = [f.tool_name for f in findings.all()]
    assert "inspect_join_viability" in names
    assert "execute_join" in names
    # Now verify that the later dataset_info step found the derived dataset
    dataset_info_finding = [f for f in findings.all() if f.tool_name == "dataset_info"]
    assert dataset_info_finding and dataset_info_finding[0].result["num_rows"] == 4


def test_execute_join_requires_viability_check(customers_orders_datasets):
    store = FindingsStore()
    registered = {}

    def registrar(name, df):
        registered[name] = df

    executor = Executor({"dataset_info": dataset_info}, findings_store=store, derived_dataset_register=registrar)

    plan = AnalysisPlan(
        id="p6",
        objective="join without inspect",
        datasets=["customers", "orders"],
        steps=[PlanStep(id="s1", tool_name="execute_join", kwargs={"left_dataset": "customers", "right_dataset": "orders", "left_on": "customer_id", "right_on": "customer_id"}, read_only=False)],
    )

    # Should succeed because executor will perform a pre-check, register derived dataset and proceed
    findings = executor.execute(plan, customers_orders_datasets)
    assert any(f.tool_name == "inspect_join_viability" for f in findings.all())
    assert any(f.tool_name == "execute_join" for f in findings.all())


def test_arbitrary_non_readonly_rejected(customers_orders_datasets):
    store = FindingsStore()
    executor = Executor({"dataset_info": dataset_info}, findings_store=store)
    plan = AnalysisPlan(
        id="p7",
        objective="mutate",
        datasets=["customers", "orders"],
        steps=[PlanStep(id="s1", tool_name="dataset_info", kwargs={"dataset_name": "customers"}, read_only=False)],
    )
    with pytest.raises(ExecutorError):
        executor.execute(plan, customers_orders_datasets)
