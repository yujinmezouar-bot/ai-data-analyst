from __future__ import annotations

import math
from typing import Any

import pandas as pd

from tools.date_utils import ALLOWED_PERIODS, add_period_column, format_period_label
from tools.groupby import MAX_GROUPS_RETURNED, aggregate_series
from tools.period_comparison import _safe_pct_change


ALLOWED_CONTRIBUTION_AGGREGATIONS = {"sum", "count"}
MAX_CONTRIBUTORS_RETURNED = min(20, MAX_GROUPS_RETURNED)
MAX_FILTER_VALUES = 50


def _finite_rounded(value: float) -> float | None:
    value = float(value)
    return round(value, 2) if math.isfinite(value) else None


def kpi_contribution_analysis(
    df: pd.DataFrame,
    date_column: str,
    metric_column: str,
    group_column: str,
    period_a: str,
    period_b: str,
    period: str = "year",
    agg_function: str = "sum",
    filter_column: str | None = None,
    filter_values: list[str | int | float] | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Decompose an additive KPI change into signed group contributions."""
    if df is None:
        return {"error": "No dataset is loaded."}

    for column, role in (
        (date_column, "date"), (metric_column, "metric"), (group_column, "group")
    ):
        if column not in df.columns:
            return {
                "error": f"Column '{column}' not found for the {role} role.",
                "available_columns": list(df.columns),
            }
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        return {"error": f"Column '{date_column}' is not a recognized date column."}
    if agg_function not in ALLOWED_CONTRIBUTION_AGGREGATIONS:
        return {
            "error": f"Aggregation function '{agg_function}' is not supported for contribution analysis.",
            "allowed_functions": sorted(ALLOWED_CONTRIBUTION_AGGREGATIONS),
        }
    if agg_function == "sum" and not pd.api.types.is_numeric_dtype(df[metric_column]):
        return {"error": f"Column '{metric_column}' must be numeric for sum aggregation."}
    if period not in ALLOWED_PERIODS:
        return {"error": f"Time period '{period}' is not supported.", "allowed_periods": sorted(ALLOWED_PERIODS)}
    if not isinstance(period_a, str) or not isinstance(period_b, str):
        return {"error": "period_a and period_b must be formatted period-label strings."}
    if (filter_column is None) != (filter_values is None):
        return {"error": "filter_column and filter_values must be supplied together."}
    if filter_column is not None:
        if filter_column not in df.columns:
            return {"error": f"Filter column '{filter_column}' not found.", "available_columns": list(df.columns)}
        if not isinstance(filter_values, list) or not filter_values:
            return {"error": "filter_values must be a non-empty list."}
        if len(filter_values) > MAX_FILTER_VALUES:
            return {"error": f"filter_values is capped at {MAX_FILTER_VALUES} values."}
        if any(not isinstance(value, (str, int, float)) or isinstance(value, bool) for value in filter_values):
            return {"error": "filter_values may contain only strings or numbers."}
    if top_n is not None:
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
            return {"error": "top_n must be a positive integer."}
        if top_n > MAX_CONTRIBUTORS_RETURNED:
            return {"error": f"top_n is capped at {MAX_CONTRIBUTORS_RETURNED}."}

    working_df = df
    filter_metadata = None
    if filter_column is not None:
        requested = [str(value) for value in filter_values]
        working_df = df[df[filter_column].astype(str).isin(requested)]
        filter_metadata = {"column": filter_column, "values": filter_values}
        if working_df.empty:
            return {"error": "No rows remain after applying the requested filter.", "filter_applied": filter_metadata}

    missing_date = int(working_df[date_column].isna().sum())
    missing_group = int(working_df[group_column].isna().sum())
    missing_metric = int(working_df[metric_column].isna().sum())
    usable_mask = (
        working_df[date_column].notna()
        & working_df[group_column].notna()
        & working_df[metric_column].notna()
    )
    usable = working_df.loc[usable_mask].copy()
    excluded_total = int((~usable_mask).sum())
    if usable.empty:
        return {"error": "No usable rows remain after excluding missing dates, groups, and metrics."}

    usable["_bucket"] = add_period_column(usable[date_column], period)
    bucket_labels = {
        format_period_label(pd.Timestamp(bucket), period): pd.Timestamp(bucket)
        for bucket in sorted(usable["_bucket"].dropna().unique())
    }
    available_periods = list(bucket_labels)
    if period_a not in bucket_labels or period_b not in bucket_labels:
        return {
            "error": "One or both requested periods do not exist in the usable data.",
            "available_periods": available_periods[-24:],
        }
    if period_a == period_b:
        return {"error": "period_a and period_b must be different."}
    if bucket_labels[period_a] >= bucket_labels[period_b]:
        return {"error": "period_a must precede period_b."}

    period_a_df = usable[usable["_bucket"] == bucket_labels[period_a]]
    period_b_df = usable[usable["_bucket"] == bucket_labels[period_b]]
    if period_a_df.empty or period_b_df.empty:
        return {"error": "Both requested periods must contain usable data."}

    try:
        values_a, _ = aggregate_series(period_a_df, group_column, metric_column, agg_function)
        values_b, _ = aggregate_series(period_b_df, group_column, metric_column, agg_function)
    except Exception as exc:
        return {"error": f"Contribution aggregation failed: {exc}"}

    all_groups = values_a.index.union(values_b.index)
    if len(all_groups) == 0:
        return {"error": "No valid groups were found in the requested periods."}

    raw_a = {str(group): float(values_a.get(group, 0.0)) for group in all_groups}
    raw_b = {str(group): float(values_b.get(group, 0.0)) for group in all_groups}
    if any(not math.isfinite(value) for value in [*raw_a.values(), *raw_b.values()]):
        return {"error": "Contribution aggregation produced non-finite values."}
    total_a = sum(raw_a.values())
    total_b = sum(raw_b.values())
    total_change = total_b - total_a
    direction = "increase" if total_change > 0 else ("decrease" if total_change < 0 else "unchanged")
    direction_sign = 1 if total_change > 0 else (-1 if total_change < 0 else 0)

    contributors = []
    for group in sorted(raw_a):
        value_a = raw_a[group]
        value_b = raw_b[group]
        change = value_b - value_a
        in_a = group in {str(item) for item in values_a.index}
        in_b = group in {str(item) for item in values_b.index}
        status = "existing" if in_a and in_b else ("new" if in_b else "disappeared")
        impact_score = change * direction_sign if direction_sign else abs(change)
        if direction_sign == 0:
            effect = "net_zero_movement"
        elif change == 0:
            effect = "neutral"
        elif impact_score > 0:
            effect = f"reinforces_{direction}"
        else:
            effect = f"offsets_{direction}"
        contribution = None if total_change == 0 else 100.0 * change / total_change
        contributors.append({
            "group": group,
            "value_a": _finite_rounded(value_a),
            "value_b": _finite_rounded(value_b),
            "absolute_change": _finite_rounded(change),
            "percentage_change": _safe_pct_change(value_a, value_b),
            "contribution_to_total_change_percentage": _finite_rounded(contribution) if contribution is not None else None,
            "effect": effect,
            "group_status": status,
            "_impact_score": impact_score,
        })

    if total_change == 0:
        ranked = sorted(contributors, key=lambda item: (-item["_impact_score"], item["group"]))
    else:
        ranked = sorted(contributors, key=lambda item: (-item["_impact_score"], item["group"]))

    limit = top_n or min(10, MAX_CONTRIBUTORS_RETURNED)
    limit = min(limit, MAX_CONTRIBUTORS_RETURNED)
    returned = ranked[:limit]
    offsets = [item for item in ranked if item["_impact_score"] < 0]
    if offsets and limit >= 2 and not any(item["_impact_score"] < 0 for item in returned):
        returned[-1] = offsets[-1]
        returned.sort(key=lambda item: (-item["_impact_score"], item["group"]))

    reinforcing = [item for item in contributors if item["_impact_score"] > 0]
    top_driver = max(reinforcing, key=lambda item: item["_impact_score"], default=None)
    largest_offset = min(offsets, key=lambda item: item["_impact_score"], default=None)
    for item in contributors:
        item.pop("_impact_score", None)
    for item in returned:
        item.pop("_impact_score", None)

    truncated = len(contributors) > len(returned)
    output = {
        "date_column": date_column,
        "metric_column": metric_column,
        "group_column": group_column,
        "period": period,
        "period_a": period_a,
        "period_b": period_b,
        "agg_function": agg_function,
        "overall": {
            "value_a": _finite_rounded(total_a),
            "value_b": _finite_rounded(total_b),
            "absolute_change": _finite_rounded(total_change),
            "percentage_change": _safe_pct_change(total_a, total_b),
            "direction": direction,
        },
        "contributors": returned,
        "top_driver": top_driver["group"] if top_driver else None,
        "largest_offset": largest_offset["group"] if largest_offset else None,
        "groups_analyzed": len(contributors),
        "groups_returned": len(returned),
        "reinforcing_group_count": len(reinforcing),
        "offsetting_group_count": len(offsets),
        "excluded_rows": {
            "missing_date": missing_date,
            "missing_group": missing_group,
            "missing_metric": missing_metric,
            "total_excluded": excluded_total,
        },
        "truncated": truncated,
    }
    if filter_metadata:
        output["filter_applied"] = filter_metadata
    if truncated:
        output["note"] = (
            f"Contributor details were truncated to {len(returned)} of {len(contributors)} groups; "
            "overall totals use all valid groups."
        )
    if total_change == 0:
        output["note"] = (
            (output.get("note", "") + " ").strip()
            + "Group gains and losses netted to zero; contribution percentages are undefined."
        ).strip()
    return output


KPI_CONTRIBUTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "kpi_contribution_analysis",
        "description": (
            "Measure signed group-level movement in an additive KPI between two grounded periods. "
            "Use to identify which products, groups, or categories declined or grew most, or which "
            "groups drove, contributed to, offset, or accounted for the total KPI change. "
            "Use groupby_analysis for group levels, percentage_change "
            "for total change only, and correlation_analysis for association. This tool identifies "
            "mathematical drivers and offsets, never causes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "Named dataset to analyze."},
                "date_column": {"type": "string", "description": "Datetime column used to define periods."},
                "metric_column": {"type": "string", "description": "Additive KPI column, such as Sales or Revenue."},
                "group_column": {"type": "string", "description": "Dimension whose contributions should be calculated."},
                "period_a": {"type": "string", "description": "Earlier formatted period label, e.g. 2024 or 2025-01."},
                "period_b": {"type": "string", "description": "Later formatted period label, e.g. 2025 or 2025-02."},
                "period": {"type": "string", "enum": sorted(ALLOWED_PERIODS), "description": "Period bucket; defaults to year."},
                "agg_function": {"type": "string", "enum": sorted(ALLOWED_CONTRIBUTION_AGGREGATIONS), "description": "Additive aggregation; defaults to sum."},
                "filter_column": {"type": ["string", "null"], "description": "Optional independent equality-filter column."},
                "filter_values": {"type": ["array", "null"], "items": {"type": ["string", "number"]}, "description": "Non-empty values matched in filter_column."},
                "top_n": {"type": ["integer", "null"], "description": f"Contributor detail limit, capped at {MAX_CONTRIBUTORS_RETURNED}."},
            },
            "required": ["date_column", "metric_column", "group_column", "period_a", "period_b"],
        },
    },
}
