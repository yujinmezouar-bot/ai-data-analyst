import json
from types import SimpleNamespace

import pandas as pd

from agent.agent import Agent


def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(
        name=name,
        arguments=json.dumps(arguments),
    ))


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class PrematureJoinProvider:
    def __init__(self):
        self.responses = [
            _message(tool_calls=[_tool_call("inspect_join_viability", {
                "left_dataset": "Walmart_Sales.csv",
                "right_dataset": "Walmart_Stores_Demo.csv",
                "left_on": "Store",
                "right_on": "Store",
            })]),
            _message("The join is safe, but the aggregation was not computed."),
            _message(tool_calls=[_tool_call("execute_join", {
                "left_dataset": "Walmart_Sales.csv",
                "right_dataset": "Walmart_Stores_Demo.csv",
                "left_on": "Store",
                "right_on": "Store",
            })]),
            _message(tool_calls=[_tool_call("groupby_analysis", {
                "dataset_name": "derived_join_1",
                "group_column": "Store_Type",
                "value_column": "Weekly_Sales",
                "agg_function": "sum",
                "sort_order": "desc",
            })]),
            _message("The aggregation is complete."),
            _message("Type A has the highest total Weekly_Sales."),
        ]

    def chat(self, messages, tools=None, tool_choice=None):
        return self.responses.pop(0)


def _datasets():
    return {
        "Walmart_Sales.csv": pd.DataFrame({
            "Store": [1, 1, 2],
            "Weekly_Sales": [100.0, 150.0, 80.0],
        }),
        "Walmart_Stores_Demo.csv": pd.DataFrame({
            "Store": [1, 2],
            "Store_Type": ["A", "B"],
            "Region": ["North", "South"],
            "Size_Category": ["Large", "Small"],
        }),
    }


def test_reactive_join_request_gets_one_bounded_chance_to_finish_downstream_work():
    agent = Agent()
    agent.llm = PrematureJoinProvider()

    result = agent.run(
        "Join Walmart_Sales.csv with Walmart_Stores_Demo.csv using Store. "
        "Then calculate total Weekly_Sales by Store_Type and identify which Store_Type "
        "has the highest total Weekly_Sales.",
        datasets=_datasets(),
        autonomous=False,
    )

    tools = [item["tool_name"] for item in result["evidence"]]
    assert tools == ["inspect_join_viability", "execute_join", "groupby_analysis"]
    grouped = result["evidence"][-1]["result"]
    assert grouped["result"] == {"A": 250.0, "B": 80.0}
    assert grouped["best_group"] == "A"
    assert "derived_join_1" in agent.derived_datasets


def test_derived_join_is_intentionally_reset_before_a_new_run():
    agent = Agent()
    agent.derived_datasets["derived_join_1"] = pd.DataFrame({"value": [1]})
    agent.llm = SimpleNamespace(chat=lambda *args, **kwargs: _message("Need a new join."))

    agent.run("Analyze the joined data.", datasets=_datasets(), autonomous=False)

    assert not agent.derived_datasets
