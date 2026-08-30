from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.datasets import build_benchmark_datasets
from evaluation.ground_truth import contribution, grouped, monthly, scalar, yearly


BENCHMARK_VERSION = "9.0"
CATEGORIES = (
    "descriptive", "ranking", "time", "change", "contribution", "correlation_outliers",
    "visualization", "multi_dataset", "autonomous", "ml_classification", "ml_regression",
    "ml_safety", "context", "fallback",
)


@dataclass(frozen=True)
class ValueExpectation:
    tool: str
    path: str
    expected: Any
    tolerance: float = 1e-6


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    category: str
    question: str
    datasets: tuple[str, ...]
    expected_mode: str | None = None
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_values: tuple[ValueExpectation, ...] = ()
    expected_warning: str | None = None
    expected_error: str | None = None
    answer_contains: tuple[str, ...] = ()
    answer_forbids: tuple[str, ...] = ()
    scripted_calls: tuple[tuple[str, dict[str, Any]], ...] = ()
    autonomous_steps: tuple[tuple[str, dict[str, Any]], ...] = ()
    fault: str | None = None
    notes: str = ""

    def validate(self) -> None:
        if not self.id or not self.question:
            raise ValueError("Benchmark cases require non-empty id and question")
        if self.category not in CATEGORIES:
            raise ValueError(f"Unknown benchmark category: {self.category}")
        if self.expected_mode not in {None, "reactive", "autonomous"}:
            raise ValueError("expected_mode must be reactive, autonomous, or null")
        if len(set(self.expected_tools) & set(self.forbidden_tools)):
            raise ValueError("A tool cannot be both required and forbidden")


def _value(tool: str, path: str, expected: Any, tolerance: float = 1e-6) -> ValueExpectation:
    return ValueExpectation(tool, path, expected, tolerance)


def benchmark_cases() -> list[BenchmarkCase]:
    data = build_benchmark_datasets()
    sales, customers = data["sales"], data["customers"]
    revenue_groups = grouped(sales, "product", "revenue")
    region_groups = grouped(sales, "region", "revenue")
    month_values = monthly(sales, "revenue")
    years = yearly(sales, "revenue")
    drivers = contribution(sales, "product", "revenue")
    region_drivers = contribution(sales, "region", "revenue")
    north_drivers = contribution(sales[sales["region"] == "North"], "product", "revenue")
    common = {"expected_mode": "reactive"}

    cases = [
        BenchmarkCase("desc_total_revenue", "descriptive", "What is total revenue?", ("sales",), expected_tools=("groupby_analysis",), scripted_calls=(("groupby_analysis", {"dataset_name":"sales","group_column":"category","value_column":"revenue","agg_function":"sum"}),), notes="Tool selection is scored; category totals sum to total revenue."),
        BenchmarkCase("desc_average_quantity", "descriptive", "What is the average quantity?", ("sales",), expected_tools=("statistics",), expected_values=(_value("statistics","mean",round(scalar(sales,"quantity","mean"),2)),), scripted_calls=(("statistics", {"dataset_name":"sales","column":"quantity"}),), **common),
        BenchmarkCase("desc_missing", "descriptive", "Which columns contain missing values?", ("sales",), expected_tools=("missing_values",), expected_values=(_value("missing_values","total_missing_values",1),), scripted_calls=(("missing_values", {"dataset_name":"sales"}),), **common),
        BenchmarkCase("desc_statistics", "descriptive", "Summarize revenue statistically.", ("sales",), expected_tools=("statistics",), expected_values=(_value("statistics","count",120),), scripted_calls=(("statistics", {"dataset_name":"sales","column":"revenue"}),), **common),

        BenchmarkCase("rank_product", "ranking", "Which product has the highest revenue?", ("sales",), expected_tools=("groupby_analysis",), expected_values=(_value("groupby_analysis","best_group",max(revenue_groups,key=revenue_groups.get)),), scripted_calls=(("groupby_analysis", {"dataset_name":"sales","group_column":"product","value_column":"revenue","agg_function":"sum"}),), **common),
        BenchmarkCase("rank_region", "ranking", "Which region has the lowest revenue?", ("sales",), expected_tools=("groupby_analysis",), expected_values=(_value("groupby_analysis","worst_group",min(region_groups,key=region_groups.get)),), scripted_calls=(("groupby_analysis", {"dataset_name":"sales","group_column":"region","value_column":"revenue","agg_function":"sum"}),), **common),
        BenchmarkCase("rank_category_average", "ranking", "Show average revenue by category.", ("sales",), expected_tools=("groupby_analysis",), scripted_calls=(("groupby_analysis", {"dataset_name":"sales","group_column":"category","value_column":"revenue","agg_function":"mean"}),), **common),
        BenchmarkCase("rank_top_three", "ranking", "Show the top 3 products by revenue.", ("sales",), expected_tools=("groupby_analysis",), scripted_calls=(("groupby_analysis", {"dataset_name":"sales","group_column":"product","value_column":"revenue","agg_function":"sum","top_n":3}),), **common),

        BenchmarkCase("time_monthly", "time", "Show monthly revenue trends.", ("sales",), expected_tools=("time_analysis",), expected_values=(_value("time_analysis","total_periods",24),), scripted_calls=(("time_analysis", {"dataset_name":"sales","date_column":"date","value_column":"revenue","period":"month","agg_function":"sum"}),), **common),
        BenchmarkCase("time_best_month", "time", "Which month had the highest revenue?", ("sales",), expected_tools=("time_analysis",), expected_values=(_value("time_analysis","best_period",max(month_values,key=month_values.get)),), scripted_calls=(("time_analysis", {"dataset_name":"sales","date_column":"date","value_column":"revenue","period":"month","agg_function":"sum"}),), **common),
        BenchmarkCase("time_region_overlay", "time", "Compare monthly revenue by region.", ("sales",), expected_mode="autonomous", expected_tools=("time_analysis",), autonomous_steps=(("time_analysis", {"dataset_name":"sales","date_column":"date","value_column":"revenue","period":"month","agg_function":"sum","group_column":"region"}),)),
        BenchmarkCase("time_latest_previous", "time", "Compare the latest month with the previous month.", ("sales",), expected_tools=("percentage_change",), scripted_calls=(("percentage_change", {"dataset_name":"sales","date_column":"date","value_column":"revenue","period":"month","agg_function":"sum"}),), **common),

        BenchmarkCase("change_years", "change", "How much did revenue change from 2024 to 2025?", ("sales",), expected_tools=("percentage_change",), expected_values=(_value("percentage_change","absolute_change",round(years[2025]-years[2024],2)),), scripted_calls=(("percentage_change", {"dataset_name":"sales","date_column":"date","value_column":"revenue","year_1":2024,"year_2":2025,"agg_function":"sum"}),), **common),
        BenchmarkCase("change_quantity", "change", "How did quantity change month over month?", ("sales",), expected_tools=("percentage_change",), scripted_calls=(("percentage_change", {"dataset_name":"sales","date_column":"date","value_column":"quantity","period":"month","agg_function":"sum"}),), **common),
        BenchmarkCase("change_zero_safe", "change", "Compare returned orders by month.", ("sales",), expected_tools=("percentage_change",), scripted_calls=(("percentage_change", {"dataset_name":"sales","date_column":"date","value_column":"returned","period":"month","agg_function":"sum"}),), **common),

        BenchmarkCase("contrib_products", "contribution", "Which products drove the revenue decline?", ("sales",), expected_mode="autonomous", expected_tools=("kpi_contribution_analysis",), expected_values=(_value("kpi_contribution_analysis","overall.absolute_change",drivers["total_change"]),_value("kpi_contribution_analysis","contributors.0.group",drivers["largest_decline"]),_value("kpi_contribution_analysis","contributors.0.effect","reinforces_decrease")), autonomous_steps=(("kpi_contribution_analysis", {"dataset_name":"sales","date_column":"date","metric_column":"revenue","group_column":"product","period_a":"2024","period_b":"2025","period":"year"}),)),
        BenchmarkCase("contrib_regions", "contribution", "Which regions contributed most to revenue growth?", ("sales",), expected_mode="autonomous", expected_tools=("kpi_contribution_analysis",), expected_values=(_value("kpi_contribution_analysis","overall.absolute_change",region_drivers["total_change"]),_value("kpi_contribution_analysis","contributors.0.group",region_drivers["largest_decline"])), autonomous_steps=(("kpi_contribution_analysis", {"dataset_name":"sales","date_column":"date","metric_column":"revenue","group_column":"region","period_a":"2024","period_b":"2025","period":"year"}),)),
        BenchmarkCase("contrib_filtered", "contribution", "Within North, which products drove the revenue decline?", ("sales",), expected_mode="autonomous", expected_tools=("kpi_contribution_analysis",), expected_values=(_value("kpi_contribution_analysis","overall.absolute_change",north_drivers["total_change"]),_value("kpi_contribution_analysis","contributors.0.group",north_drivers["largest_decline"])), autonomous_steps=(("kpi_contribution_analysis", {"dataset_name":"sales","date_column":"date","metric_column":"revenue","group_column":"product","period_a":"2024","period_b":"2025","period":"year","filter_column":"region","filter_values":["North"]}),)),

        BenchmarkCase("corr_price_quantity", "correlation_outliers", "Is unit price correlated with quantity?", ("sales",), expected_tools=("correlation_analysis",), answer_forbids=("causes",), scripted_calls=(("correlation_analysis", {"dataset_name":"sales","columns":["unit_price","quantity"]}),), **common),
        BenchmarkCase("corr_revenue", "correlation_outliers", "What is most correlated with revenue?", ("sales",), expected_tools=("correlation_analysis",), scripted_calls=(("correlation_analysis", {"dataset_name":"sales","column":"revenue"}),), **common),
        BenchmarkCase("outlier_revenue", "correlation_outliers", "Are there extreme revenue values?", ("sales",), expected_tools=("outlier_analysis",), expected_values=(_value("outlier_analysis","outlier_count",1),), scripted_calls=(("outlier_analysis", {"dataset_name":"sales","column":"revenue"}),), **common),

        BenchmarkCase("viz_monthly", "visualization", "Plot monthly revenue.", ("sales",), expected_tools=("create_visualization",), scripted_calls=(("create_visualization", {"dataset_name":"sales","chart_type":"line","x_column":"date","y_column":"revenue","agg_function":"sum","period":"month"}),), **common),
        BenchmarkCase("viz_category", "visualization", "Create a bar chart comparing revenue by category.", ("sales",), expected_tools=("create_visualization",), scripted_calls=(("create_visualization", {"dataset_name":"sales","chart_type":"bar","x_column":"category","y_column":"revenue","agg_function":"sum"}),), **common),
        BenchmarkCase("viz_distribution", "visualization", "Show the distribution of quantity.", ("sales",), expected_tools=("create_visualization",), scripted_calls=(("create_visualization", {"dataset_name":"sales","chart_type":"histogram","x_column":"quantity"}),), **common),

        BenchmarkCase("multi_relationship", "multi_dataset", "Discover relationships between the uploaded datasets.", ("sales","customers","products"), expected_tools=("discover_relationships",), scripted_calls=(("discover_relationships", {}),), **common),
        BenchmarkCase("multi_safe_join", "multi_dataset", "Join sales with customers using customer_id.", ("sales","customers"), expected_tools=("execute_join",), expected_values=(_value("execute_join","cardinality","N:1"),), scripted_calls=(("execute_join", {"left_dataset":"sales","right_dataset":"customers","left_on":"customer_id","right_on":"customer_id"}),), **common),
        BenchmarkCase("multi_unsafe_join", "multi_dataset", "Can sales be joined to events on customer_id safely?", ("sales","events"), expected_tools=("inspect_join_viability",), expected_values=(_value("inspect_join_viability","safe_to_join",False),), scripted_calls=(("inspect_join_viability", {"left_dataset":"sales","right_dataset":"events","left_on":"customer_id","right_on":"customer_id"}),), **common),
        BenchmarkCase("multi_join_analysis", "multi_dataset", "Compare customer segment revenue over time after joining sales and customers.", ("sales","customers"), expected_mode="autonomous", expected_tools=("execute_join","time_analysis"), autonomous_steps=(("execute_join", {"left_dataset":"sales","right_dataset":"customers","left_on":"customer_id","right_on":"customer_id"}),("time_analysis", {"dataset_name":"derived_join_1","date_column":"date","value_column":"revenue","period":"month","agg_function":"sum","group_column":"segment"}))),

        BenchmarkCase(
            "auto_decline_why", "autonomous",
            "Which products declined the most from 2024 to 2025, and why, based on observed factors?", ("sales",),
            expected_mode="autonomous",
            expected_tools=("kpi_contribution_analysis",),
            expected_values=(
                _value("kpi_contribution_analysis", "period_a", drivers["period_a"]),
                _value("kpi_contribution_analysis", "period_b", drivers["period_b"]),
                _value("kpi_contribution_analysis", "contributors.0.group", drivers["largest_decline"]),
                _value("kpi_contribution_analysis", "contributors.0.value_a", drivers["leading_value_a"]),
                _value("kpi_contribution_analysis", "contributors.0.value_b", drivers["leading_value_b"]),
                _value("kpi_contribution_analysis", "contributors.0.absolute_change", drivers["leading_absolute_change"]),
                _value("kpi_contribution_analysis", "contributors.0.percentage_change", drivers["leading_percentage_change"], 1e-2),
                _value("kpi_contribution_analysis", "contributors.0.effect", drivers["leading_effect"]),
                _value("kpi_contribution_analysis", "overall.absolute_change", drivers["total_change"]),
                _value("kpi_contribution_analysis", "overall.direction", drivers["direction"]),
            ),
            autonomous_steps=(
                ("time_analysis", {"dataset_name":"sales","date_column":"date","value_column":"revenue","period":"year","agg_function":"sum","group_column":"product"}),
                ("kpi_contribution_analysis", {"dataset_name":"sales","date_column":"date","metric_column":"revenue","group_column":"product","period_a":"2024","period_b":"2025"}),
            ),
        ),
        BenchmarkCase("auto_relationship_time", "autonomous", "Analyze the relationship between price and quantity over time.", ("sales",), expected_mode="autonomous", expected_tools=("correlation_analysis","time_analysis"), autonomous_steps=(("correlation_analysis", {"dataset_name":"sales","columns":["unit_price","quantity"]}),("time_analysis", {"dataset_name":"sales","date_column":"date","value_column":"unit_price","period":"month","agg_function":"mean"}))),
        BenchmarkCase("auto_ranking_time", "autonomous", "Which regions had the biggest revenue decline over time?", ("sales",), expected_mode="autonomous", expected_tools=("time_analysis",), autonomous_steps=(("time_analysis", {"dataset_name":"sales","date_column":"date","value_column":"revenue","period":"year","agg_function":"sum","group_column":"region"}),)),
        BenchmarkCase("auto_visual_veto", "autonomous", "Plot monthly revenue and explain it.", ("sales",), expected_mode="reactive", expected_tools=("create_visualization",), scripted_calls=(("create_visualization", {"dataset_name":"sales","chart_type":"line","x_column":"date","y_column":"revenue","agg_function":"sum","period":"month"}),)),

        BenchmarkCase("ml_churn", "ml_classification", "Build a classification model to predict churn.", ("customers",), expected_tools=("train_ml_model",), expected_values=(_value("train_ml_model","task_type","classification"),_value("train_ml_model","target_column","churn")), scripted_calls=(("train_ml_model", {"dataset_name":"customers","target_column":"churn","task_type":"classification"}),), **common),
        BenchmarkCase("ml_returns", "ml_classification", "Build a classification model to predict returned orders.", ("sales",), expected_tools=("train_ml_model",), scripted_calls=(("train_ml_model", {"dataset_name":"sales","target_column":"returned","task_type":"classification"}),), **common),
        BenchmarkCase("ml_unseen_customer", "ml_classification", "Predict returns for unseen customers using customer_id as the group.", ("sales",), expected_tools=("train_ml_model",), expected_values=(_value("train_ml_model","split.group_aware",True),_value("train_ml_model","split.group_overlap_count",0)), scripted_calls=(("train_ml_model", {"dataset_name":"sales","target_column":"returned","task_type":"classification","group_column":"customer_id"}),), **common),

        BenchmarkCase("ml_revenue_regression", "ml_regression", "Build a regression model to predict revenue.", ("sales",), expected_tools=("train_ml_model",), expected_values=(_value("train_ml_model","task_type","regression"),), scripted_calls=(("train_ml_model", {"dataset_name":"sales","target_column":"revenue","task_type":"regression"}),), **common),
        BenchmarkCase("ml_price_regression", "ml_regression", "Predict unit_price using regression.", ("sales",), expected_tools=("train_ml_model",), scripted_calls=(("train_ml_model", {"dataset_name":"sales","target_column":"unit_price","task_type":"regression"}),), **common),

        BenchmarkCase("ml_ambiguous", "ml_safety", "Build a model for this dataset.", ("sales",), expected_tools=(), forbidden_tools=("train_ml_model",), answer_contains=("target", "task"), **common),
        BenchmarkCase("ml_explicit_id", "ml_safety", "Use order_id to predict returned.", ("sales",), expected_tools=("train_ml_model",), expected_error="identifier-like", scripted_calls=(("train_ml_model", {"dataset_name":"sales","target_column":"returned","task_type":"classification","feature_columns":["order_id"]}),), **common),
        BenchmarkCase("ml_small", "ml_safety", "Predict churn in this small dataset.", ("small_customers",), expected_tools=("train_ml_model",), expected_error="At least 30", scripted_calls=(("train_ml_model", {"dataset_name":"small_customers","target_column":"churn","task_type":"classification"}),), **common),
        BenchmarkCase("ml_repeated_warning", "ml_safety", "Predict returned using a row-level split.", ("sales",), expected_tools=("train_ml_model",), expected_warning="Repeated entities", scripted_calls=(("train_ml_model", {"dataset_name":"sales","target_column":"returned","task_type":"classification"}),), **common),
        BenchmarkCase("ml_group_temporal", "ml_safety", "Use temporal and customer-group isolation to predict returns.", ("sales",), expected_tools=("train_ml_model",), expected_error="grouped-temporal", scripted_calls=(("train_ml_model", {"dataset_name":"sales","target_column":"returned","task_type":"classification","split_strategy":"temporal","time_column":"date","group_column":"customer_id"}),), **common),

        BenchmarkCase("context_those", "context", "Compare those stores by month.", ("sales",), expected_mode="reactive", answer_contains=("context",), notes="Current router deliberately vetoes context-dependent autonomy."),
        BenchmarkCase("context_previous", "context", "Analyze the previous result.", ("sales",), expected_mode="reactive", answer_contains=("context",), notes="No structured prior result is supplied in this single-turn case."),

        BenchmarkCase("fallback_planner", "fallback", "Which products declined the most over time?", ("sales",), expected_mode="autonomous", answer_contains=("fallback",), fault="malformed_planner"),
        BenchmarkCase("fallback_synthesis", "fallback", "Which products drove the revenue decline?", ("sales",), expected_mode="autonomous", expected_tools=("kpi_contribution_analysis",), fault="synthesis_failure", autonomous_steps=(("kpi_contribution_analysis", {"dataset_name":"sales","date_column":"date","metric_column":"revenue","group_column":"product","period_a":"2024","period_b":"2025"}),)),
        BenchmarkCase("fallback_tool_error", "fallback", "What is the average missing_column?", ("sales",), expected_mode="reactive", expected_tools=("statistics",), expected_error="not found", scripted_calls=(("statistics", {"dataset_name":"sales","column":"missing_column"}),)),
    ]
    for case in cases:
        case.validate()
    return cases
