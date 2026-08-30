from __future__ import annotations

from typing import Any

import pandas as pd


def scalar(dataframe: pd.DataFrame, column: str, aggregation: str) -> float:
    return float(dataframe[column].agg(aggregation))


def grouped(dataframe: pd.DataFrame, group: str, value: str, aggregation: str = "sum") -> dict[str, float]:
    result = dataframe.groupby(group)[value].agg(aggregation)
    return {str(key): float(value) for key, value in result.items()}


def monthly(dataframe: pd.DataFrame, value: str, aggregation: str = "sum") -> dict[str, float]:
    result = dataframe.groupby(dataframe["date"].dt.to_period("M"))[value].agg(aggregation)
    return {str(key): float(value) for key, value in result.items()}


def yearly(dataframe: pd.DataFrame, value: str, aggregation: str = "sum") -> dict[int, float]:
    result = dataframe.groupby(dataframe["date"].dt.year)[value].agg(aggregation)
    return {int(key): float(value) for key, value in result.items()}


def contribution(dataframe: pd.DataFrame, group: str, value: str) -> dict[str, Any]:
    values = dataframe.groupby([dataframe["date"].dt.year, group])[value].sum().unstack(fill_value=0)
    period_a, period_b = 2024, 2025
    values_a = values.loc[period_a]
    values_b = values.loc[period_b]
    changes = values_b.sub(values_a, fill_value=0)
    largest_decline = changes.idxmin()
    leading_value_a = float(values_a.get(largest_decline, 0.0))
    leading_value_b = float(values_b.get(largest_decline, 0.0))
    leading_change = float(changes[largest_decline])
    total_change = float(changes.sum())
    direction = "decrease" if total_change < 0 else ("increase" if total_change > 0 else "unchanged")
    return {
        "period_a": str(period_a),
        "period_b": str(period_b),
        "total_change": total_change,
        "direction": direction,
        "changes": {str(key): float(item) for key, item in changes.items()},
        "largest_decline": str(largest_decline),
        "largest_growth": str(changes.idxmax()),
        "leading_value_a": leading_value_a,
        "leading_value_b": leading_value_b,
        "leading_absolute_change": leading_change,
        "leading_percentage_change": (
            None if leading_value_a == 0
            else round(100.0 * leading_change / leading_value_a, 2)
        ),
        "leading_effect": f"reinforces_{direction}" if total_change and leading_change * total_change > 0 else "offset",
    }
