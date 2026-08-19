from typing import Any

import pandas as pd

from tools.date_utils import ALLOWED_PERIODS, add_period_column, format_period_label
from tools.groupby import ALLOWED_AGG_FUNCTIONS, aggregate_series


MAX_PERIODS_RETURNED = 24  # e.g. two years of monthly data, kept compact


def _safe_pct_change(previous: float, current: float) -> float | None:
    """
    Percentage change that never raises or returns inf/NaN. If the
    previous value is zero, percentage change is undefined -- return
    None rather than a misleading infinite number.
    """
    if previous == 0:
        return None
    return round(100 * (current - previous) / previous, 2)


def percentage_change(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    period: str = "month",
    agg_function: str = "sum",
    year_1: int | None = None,
    year_2: int | None = None,
    group_column: str | None = None,
    filter_values: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare a numeric column across time periods.

    Two modes:
    - Explicit year comparison (year_1 vs year_2), e.g. "how did sales
      change from 2024 to 2025?".
    - Period-over-period comparison (default), e.g. "compare this month
      with the previous month" / "which month had the largest increase?".
      Reuses the same period bucketing as time_analysis so results stay
      consistent across tools.
    """
    if df is None:
        return {"error": "No dataset is loaded."}

    period = period or "month"
    agg_function = agg_function or "sum"

    if date_column not in df.columns:
        return {"error": f"Column '{date_column}' not found in the dataset.", "available_columns": list(df.columns)}

    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        return {"error": f"Column '{date_column}' is not a recognized date column. Use dataset_info to see detected date columns."}

    if value_column not in df.columns:
        return {"error": f"Column '{value_column}' not found in the dataset.", "available_columns": list(df.columns)}

    if agg_function != "count" and not pd.api.types.is_numeric_dtype(df[value_column]):
        return {"error": f"Column '{value_column}' is not numeric, so '{agg_function}' cannot be computed on it."}

    if agg_function not in ALLOWED_AGG_FUNCTIONS:
        return {"error": f"Aggregation function '{agg_function}' is not supported.", "allowed_functions": sorted(ALLOWED_AGG_FUNCTIONS)}

    if period not in ALLOWED_PERIODS:
        return {"error": f"Time period '{period}' is not supported.", "allowed_periods": sorted(ALLOWED_PERIODS)}

    if group_column is not None and group_column not in df.columns:
        return {"error": f"Column '{group_column}' not found in the dataset.", "available_columns": list(df.columns)}

    working_df = df.dropna(subset=[date_column, value_column]).copy()

    if group_column is not None and filter_values:
        working_df = working_df[working_df[group_column].astype(str).isin([str(v) for v in filter_values])]

    if working_df.empty:
        return {"error": "No rows remain after applying the requested filters."}

    # ------------------------------------------------------------
    # Mode 1: explicit year vs year comparison.
    # ------------------------------------------------------------
    if year_1 is not None or year_2 is not None:
        if year_1 is None or year_2 is None:
            return {"error": "Both year_1 and year_2 must be provided to compare specific years."}

        subset_1 = working_df[working_df[date_column].dt.year == year_1]
        subset_2 = working_df[working_df[date_column].dt.year == year_2]

        if subset_1.empty:
            return {"error": f"No data found for year {year_1}."}
        if subset_2.empty:
            return {"error": f"No data found for year {year_2}."}

        try:
            value_1 = float(subset_1[value_column].agg(agg_function))
            value_2 = float(subset_2[value_column].agg(agg_function))
        except Exception as e:
            return {"error": f"Failed to aggregate values: {e}"}

        abs_diff = round(value_2 - value_1, 2)
        mean_val = (value_1 + value_2) / 2.0
        pct_diff = round(100.0 * abs(abs_diff) / mean_val, 2) if mean_val != 0 else None
        comp_summary = "increased" if value_2 > value_1 else ("decreased" if value_2 < value_1 else "unchanged")

        output = {
            "date_column": date_column,
            "value_column": value_column,
            "agg_function": agg_function,
            "previous_period": str(year_1),
            "current_period": str(year_2),
            "previous_value": round(value_1, 2),
            "current_value": round(value_2, 2),
            "absolute_change": abs_diff,
            "percentage_change": _safe_pct_change(value_1, value_2),
            "percentage_difference": pct_diff,
            "comparison_summary": comp_summary,
        }
        if group_column is not None and filter_values:
            output["filter_applied"] = {"column": group_column, "values": filter_values}
        return output

    # ------------------------------------------------------------
    # Mode 2: period-over-period comparison, reusing the same
    # bucketing logic as time_analysis.
    # ------------------------------------------------------------
    working_df["_bucket"] = add_period_column(working_df[date_column], period)

    try:
        series, _ = aggregate_series(working_df, "_bucket", value_column, agg_function)
    except Exception as e:
        return {"error": f"Period aggregation failed: {e}"}

    labeled = [
        (format_period_label(bucket, period), None if pd.isna(value) else float(value))
        for bucket, value in series.items()
    ]
    labeled = [item for item in labeled if item[1] is not None]

    if len(labeled) < 2:
        return {"error": f"Not enough periods with data to compare (found {len(labeled)}). Try a coarser period."}

    changes = []
    for (prev_label, prev_value), (curr_label, curr_value) in zip(labeled, labeled[1:]):
        changes.append({
            "previous_period": prev_label,
            "current_period": curr_label,
            "previous_value": round(prev_value, 2),
            "current_value": round(curr_value, 2),
            "absolute_change": round(curr_value - prev_value, 2),
            "percentage_change": _safe_pct_change(prev_value, curr_value),
        })

    latest_change = changes[-1]

    changes_with_pct = [c for c in changes if c["percentage_change"] is not None]
    largest_increase = max(changes_with_pct, key=lambda c: c["percentage_change"], default=None)
    largest_decrease = min(changes_with_pct, key=lambda c: c["percentage_change"], default=None)

    truncated = len(changes) > MAX_PERIODS_RETURNED
    returned_changes = changes[-MAX_PERIODS_RETURNED:] if truncated else changes

    first_val = changes[0]["previous_value"]
    last_val = changes[-1]["current_value"]

    output = {
        "date_column": date_column,
        "value_column": value_column,
        "period": period,
        "agg_function": agg_function,
        "total_periods_compared": len(changes),
        "latest_change": latest_change,
        "largest_increase": largest_increase,
        "largest_decrease": largest_decrease,
        "overall_change": round(last_val - first_val, 2),
        "overall_percentage_change": _safe_pct_change(first_val, last_val),
        "changes": returned_changes,
    }
    if truncated:
        output["note"] = (
            f"Showing the most recent {MAX_PERIODS_RETURNED} of {len(changes)} period-over-period "
            "changes. largest_increase/largest_decrease were computed over the full range."
        )
    if group_column is not None and filter_values:
        output["filter_applied"] = {"column": group_column, "values": filter_values}
    return output


PERCENTAGE_CHANGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "percentage_change",
        "description": (
            "Compare a numeric column across time periods and compute absolute and percentage "
            "change. Use this for questions like 'how did sales change from 2024 to 2025?', "
            "'what is the percentage increase in sales?', 'compare this month with the previous "
            "month', or 'which month had the largest increase?'. For a specific two-year "
            "comparison, set year_1 and year_2. Otherwise, set 'period' to get period-over-period "
            "changes (the result includes the latest change plus the largest increase/decrease "
            "across the whole range). Division by zero (previous value of 0) is handled safely "
            "and returns a null percentage_change rather than an error."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_column": {"type": "string", "description": "The datetime column to bucket by, e.g. 'Date'."},
                "value_column": {"type": "string", "description": "The numeric column to compare, e.g. 'Weekly_Sales'."},
                "period": {"type": ["string", "null"], "enum": sorted(ALLOWED_PERIODS) + [None], "description": "Time bucket size for period-over-period comparison. Defaults to 'month' if null or omitted. Ignored if year_1/year_2 are set."},
                "agg_function": {"type": ["string", "null"], "enum": sorted(ALLOWED_AGG_FUNCTIONS) + [None], "description": "Aggregation within each period. Defaults to 'sum' if null or omitted."},
                "year_1": {"type": ["integer", "null"], "description": "Earlier/baseline year for an explicit year-vs-year comparison, e.g. 2024. Omit or set null for period-over-period mode."},
                "year_2": {"type": ["integer", "null"], "description": "Later/comparison year for an explicit year-vs-year comparison, e.g. 2025. Omit or set null for period-over-period mode."},
                "group_column": {"type": ["string", "null"], "description": "Optional categorical column to restrict via filter_values. Set to null (or omit) for no grouping."},
                "filter_values": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Restrict group_column to specific values. Set to null (or omit) for no filtering."},
            },
            "required": ["date_column", "value_column"],
        },
    },
}
