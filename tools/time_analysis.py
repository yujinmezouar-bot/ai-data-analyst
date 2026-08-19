from typing import Any

import pandas as pd

from tools.date_utils import ALLOWED_PERIODS, add_period_column, format_period_label
from tools.groupby import ALLOWED_AGG_FUNCTIONS, MAX_GROUPS_RETURNED, aggregate_series


def time_analysis(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    period: str = "month",
    agg_function: str = "mean",
    year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    group_column: str | None = None,
    filter_values: list[str] | None = None,
) -> dict[str, Any]:
    """
    Aggregate a numeric column over time, bucketed into day/week/month/
    quarter/year periods. All calculations happen here in Pandas; results
    are always returned in chronological order, and best/worst periods
    are computed directly so the LLM never has to infer them.
    """
    if df is None:
        return {"error": "No dataset is loaded."}

    if date_column not in df.columns:
        return {"error": f"Column '{date_column}' not found in the dataset.", "available_columns": list(df.columns)}

    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        return {"error": f"Column '{date_column}' is not a recognized date column. Use dataset_info to see detected date columns."}

    if value_column not in df.columns:
        return {"error": f"Column '{value_column}' not found in the dataset.", "available_columns": list(df.columns)}

    if agg_function != "count" and not pd.api.types.is_numeric_dtype(df[value_column]):
        return {
            "error": (
                f"Column '{value_column}' is not numeric, so '{agg_function}' cannot be "
                "computed on it. Try 'count' instead, or pick a numeric column."
            )
        }

    if period not in ALLOWED_PERIODS:
        return {"error": f"Time period '{period}' is not supported.", "allowed_periods": sorted(ALLOWED_PERIODS)}

    if agg_function not in ALLOWED_AGG_FUNCTIONS:
        return {"error": f"Aggregation function '{agg_function}' is not supported.", "allowed_functions": sorted(ALLOWED_AGG_FUNCTIONS)}

    if group_column is not None and group_column not in df.columns:
        return {"error": f"Column '{group_column}' not found in the dataset.", "available_columns": list(df.columns)}

    working_df = df.dropna(subset=[date_column, value_column]).copy()

    if year is not None:
        working_df = working_df[working_df[date_column].dt.year == year]

    if start_date is not None:
        try:
            start_ts = pd.to_datetime(start_date)
            working_df = working_df[working_df[date_column] >= start_ts]
        except Exception:
            return {"error": f"Could not parse start_date '{start_date}'. Use format YYYY-MM-DD."}

    if end_date is not None:
        try:
            end_ts = pd.to_datetime(end_date)
            working_df = working_df[working_df[date_column] <= end_ts]
        except Exception:
            return {"error": f"Could not parse end_date '{end_date}'. Use format YYYY-MM-DD."}

    if group_column is not None and filter_values:
        working_df = working_df[working_df[group_column].astype(str).isin([str(v) for v in filter_values])]

    if working_df.empty:
        return {
            "error": "No rows remain after applying the requested filters.",
            "date_column": date_column, "year": year, "start_date": start_date, "end_date": end_date,
        }

    working_df["_bucket"] = add_period_column(working_df[date_column], period)

    try:
        if group_column is not None:
            distinct_groups = working_df[group_column].nunique()

            if distinct_groups > MAX_GROUPS_RETURNED:
                return {
                    "error": (
                        f"'{group_column}' has {distinct_groups} distinct values -- too many to "
                        f"break down by {period} at once. Restrict with filter_values, or use "
                        "groupby_analysis with top_n first to pick a smaller set."
                    )
                }

            grouped = (
                working_df.groupby(["_bucket", group_column])[value_column]
                .agg(agg_function)
                .sort_index()
            )

            result: dict[str, dict[str, Any]] = {}
            for (bucket, group_key), value in grouped.items():
                label = format_period_label(bucket, period)
                result.setdefault(label, {})
                result[label][str(group_key)] = None if pd.isna(value) else round(float(value), 2)

            output = {
                "date_column": date_column,
                "value_column": value_column,
                "group_column": group_column,
                "period": period,
                "agg_function": agg_function,
                "result": result,
                "total_periods": len(result),
            }
            if filter_values:
                output["filter_applied"] = {"column": group_column, "values": filter_values}
            return output

        else:
            series, _ = aggregate_series(working_df, "_bucket", value_column, agg_function)

            result = {}
            for bucket, value in series.items():
                label = format_period_label(bucket, period)
                result[label] = None if pd.isna(value) else round(float(value), 2)

            numeric_items = [(k, v) for k, v in result.items() if v is not None]
            best_period = max(numeric_items, key=lambda kv: kv[1])[0] if numeric_items else None
            worst_period = min(numeric_items, key=lambda kv: kv[1])[0] if numeric_items else None

            trend_direction = "insufficient_data"
            overall_change = None
            overall_pct_change = None

            if len(numeric_items) >= 2:
                vals = [v for _, v in numeric_items]
                first_val, last_val = vals[0], vals[-1]
                overall_change = round(last_val - first_val, 2)
                if first_val != 0:
                    overall_pct_change = round(100.0 * (last_val - first_val) / first_val, 2)

                if max(vals) == min(vals):
                    trend_direction = "stable"
                elif all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
                    trend_direction = "strictly_increasing"
                elif all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
                    trend_direction = "strictly_decreasing"
                elif last_val > first_val:
                    trend_direction = "increasing"
                elif last_val < first_val:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "fluctuating"

            return {
                "date_column": date_column,
                "value_column": value_column,
                "period": period,
                "agg_function": agg_function,
                "result": result,
                "best_period": best_period,
                "worst_period": worst_period,
                "total_periods": len(numeric_items),
                "trend_direction": trend_direction,
                "overall_change": overall_change,
                "overall_percentage_change": overall_pct_change,
            }

    except Exception as e:
        return {"error": f"Time analysis failed: {e}"}


TIME_ANALYSIS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "time_analysis",
        "description": (
            "Analyze how a numeric column changes over time, bucketed by day, week, "
            "month, quarter, or year. Use this for trends, monthly/quarterly/yearly "
            "sales, sales over time, which period was highest/lowest/best/worst, a "
            "specific year, a date range ('from January to June'), comparing years, "
            "or comparing specific categories over time ('compare those stores by "
            "month'). Python computes best_period and worst_period exactly -- read "
            "them from the result rather than guessing. Results are always in "
            "chronological order."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_column": {"type": "string", "description": "The datetime column to bucket by, e.g. 'Date'."},
                "value_column": {"type": "string", "description": "The numeric column to aggregate, e.g. 'Weekly_Sales'."},
                "period": {"type": "string", "enum": sorted(ALLOWED_PERIODS), "description": "Time bucket size. Defaults to 'month'."},
                "agg_function": {"type": "string", "enum": sorted(ALLOWED_AGG_FUNCTIONS), "description": "Aggregation within each period. Defaults to 'mean'."},
                "year": {"type": "integer", "description": "Restrict to a single calendar year, e.g. 2024."},
                "start_date": {"type": "string", "description": "Restrict to dates on/after this date, format YYYY-MM-DD."},
                "end_date": {"type": "string", "description": "Restrict to dates on/before this date, format YYYY-MM-DD."},
                "group_column": {"type": "string", "description": "Optional secondary categorical column, e.g. 'Store', for 'compare those stores by month'."},
                "filter_values": {"type": "array", "items": {"type": "string"}, "description": "Restrict group_column to specific values, e.g. store names identified earlier."},
            },
            "required": ["date_column", "value_column"],
        },
    },
}
