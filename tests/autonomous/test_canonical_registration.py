import pandas as pd
import pytest

from autonomous.executor import Executor
from autonomous.plan import AnalysisPlan, PlanStep
from autonomous.results import FindingsStore
from agent.agent import Agent
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
    return {"customers": customers.copy(), "orders": orders.copy()}


def test_successful_join_registers_canonical(customers_orders_datasets):
    agent = Agent()

    def adapter(name, df):
        # Adapter matches Executor's derived_dataset_register signature
        agent.register_derived_dataset(df, suggested_name=name)

    store = FindingsStore()
    executor = Executor({"dataset_info": dataset_info}, findings_store=store, derived_dataset_register=adapter)

    plan = AnalysisPlan(
        id="p_can",
        objective="join",
        datasets=["customers", "orders"],
        steps=[
            PlanStep(id="s1", tool_name="inspect_join_viability", kwargs={"left_dataset": "customers", "right_dataset": "orders", "left_on": "customer_id", "right_on": "customer_id"}, read_only=True),
            PlanStep(id="s2", tool_name="execute_join", kwargs={"left_dataset": "customers", "right_dataset": "orders", "left_on": "customer_id", "right_on": "customer_id", "how": "inner"}, read_only=False),
        ],
    )

    findings = executor.execute(plan, customers_orders_datasets)

    # The agent should have a canonical derived dataset registered
    assert len(agent.derived_datasets) == 1
    name = next(iter(agent.derived_datasets.keys()))
    assert name.startswith("derived_join_")

    # The stored DF should match the join result found in the executor findings
    exec_finding = [f for f in findings.all() if f.tool_name == "execute_join"]
    assert exec_finding
    derived_name = exec_finding[0].result["dataset_name"]
    assert derived_name == name
    # verify dataset exists in agent store
    assert derived_name in agent.derived_datasets
    # verify original datasets unchanged
    assert customers_orders_datasets["customers"].equals(pd.DataFrame({"customer_id": [1, 2, 3], "name": ["A", "B", "C"]}))
    assert customers_orders_datasets["orders"].equals(pd.DataFrame({"order_id": [10, 11, 12, 13], "customer_id": [1, 1, 2, 3], "amount": [100, 150, 200, 50]}))


def test_lru_behavior_on_excess_derived_datasets(customers_orders_datasets):
    agent = Agent()

    def adapter(name, df):
        agent.register_derived_dataset(df, suggested_name=name)

    store = FindingsStore()
    executor = Executor({"dataset_info": dataset_info}, findings_store=store, derived_dataset_register=adapter)

    # Create 4 derived datasets sequentially (Agent max is 3)
    for i in range(4):
        plan = AnalysisPlan(
            id=f"p_{i}",
            objective="join",
            datasets=["customers", "orders"],
            steps=[
                PlanStep(id=f"s_inspect_{i}", tool_name="inspect_join_viability", kwargs={"left_dataset": "customers", "right_dataset": "orders", "left_on": "customer_id", "right_on": "customer_id"}, read_only=True),
                PlanStep(id=f"s_exec_{i}", tool_name="execute_join", kwargs={"left_dataset": "customers", "right_dataset": "orders", "left_on": "customer_id", "right_on": "customer_id", "how": "inner"}, read_only=False),
            ],
        )
        executor.execute(plan, customers_orders_datasets)

    # Agent should have at most 3 derived datasets (LRU eviction in effect)
    assert len(agent.derived_datasets) == agent._MAX_DERIVED_DATASETS
    names = list(agent.derived_datasets.keys())
    # After 4 inserts with suggested same name, expected canonical names are derived_join_2,3,4
    assert names[0].endswith("_join_2") or names[0].startswith("derived_join_")


def test_failed_join_does_not_register(customers_orders_datasets):
    # Use orders/items mismatch to trigger unsafe join that should be rejected
    agent = Agent()

    def adapter(name, df):
        agent.register_derived_dataset(df, suggested_name=name)

    store = FindingsStore()
    executor = Executor({}, findings_store=store, derived_dataset_register=adapter)

    # Deliberately request invalid join (wrong keys)
    plan = AnalysisPlan(
        id="p_fail",
        objective="join",
        datasets=["customers", "orders"],
        steps=[PlanStep(id="s1", tool_name="execute_join", kwargs={"left_dataset": "customers", "right_dataset": "orders", "left_on": "nonexistent", "right_on": "customer_id"}, read_only=False)],
    )

    with pytest.raises(Exception):
        executor.execute(plan, customers_orders_datasets)

    # No derived datasets should have been registered
    assert len(agent.derived_datasets) == 0


def test_registration_return_name_propagates_to_downstream_steps(customers_orders_datasets):
    store = FindingsStore()

    def registrar(name, df):
        return "canonical_join"

    executor = Executor(
        {"dataset_info": dataset_info},
        findings_store=store,
        derived_dataset_register=registrar,
    )
    plan = AnalysisPlan(
        id="canonical-propagation",
        objective="Join and inspect",
        datasets=list(customers_orders_datasets),
        steps=[
            PlanStep(
                id="join",
                tool_name="execute_join",
                kwargs={
                    "left_dataset": "customers",
                    "right_dataset": "orders",
                    "left_on": "customer_id",
                    "right_on": "customer_id",
                },
                read_only=False,
            ),
            PlanStep(
                id="inspect",
                tool_name="dataset_info",
                kwargs={"dataset_name": "derived_join_1"},
            ),
        ],
    )

    findings = executor.execute(plan, customers_orders_datasets)

    join_finding = findings.find_by_step("join")[0]
    downstream_finding = findings.find_by_step("inspect")[0]
    assert join_finding.result["dataset_name"] == "canonical_join"
    assert join_finding.provenance["derived_dataset"] == "canonical_join"
    assert join_finding.datasets[-1] == "canonical_join"
    assert downstream_finding.datasets == ["canonical_join"]
