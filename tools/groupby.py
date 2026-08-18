from typing import Any

import pandas as pd


ALLOWED_AGG_FUNCTIONS = {"mean", "sum", "count", "min", "max", "median", "std"}

# Safety cap: if a groupby produces more groups than this and the caller
# didn't ask for a specific top_n, we truncate rather than send a huge
# payload to the LLM.
MAX_GROUPS_RETURNED = 50


def aggregate_series(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    agg_function: str,
) -> tuple[pd.Series, bool]:
    """
    Core aggregation logic shared by groupby_analysis and
    create_visualization, so there is one place that decides how
    grouping + sorting works for both categorical and datetime columns.

    Returns (aggregated_series, is_datetime_group).
    """
    working_df = df.copy()
    is_datetime = pd.api.types.is_datetime64_any_dtype(working_df[group_column])

    if is_datetime:
        working_df = working_df.dropna(subset=[group_column])
        grouped = working_df.groupby(group_column)[value_column].agg(agg_function).sort_index()
    else:
        grouped = working_df.groupby(group_column)[value_column].agg(agg_function)

    return grouped, is_datetime


def apply_top_n(result: dict[str, Any], top_n: int, sort_order: str = "desc") -> dict[str, Any]:
    """
    Sort a {group: value} dict by value and keep only the top_n entries.
    Entries with a None value are always placed last.
    """
    items = list(result.items())
    numeric_items = [(k, v) for k, v in items if v is not None]
    none_items = [(k, v) for k, v in items if v is None]

    numeric_items.sort(key=lambda kv: kv[1], reverse=(sort_order != "asc"))

    return dict((numeric_items + none_items)[:top_n])


def groupby_analysis(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    agg_function: str = "mean",
    filter_values: list[str] | None = None,
    top_n: int | None = None,
    sort_order: str | None = None,
) -> dict[str, Any]:
    """
    Group the dataset by group_column and aggregate value_column.
    Supports categorical and datetime group columns, optional
    restriction to specific category values, and optional top-N sorting.
    """
    if df is None:
        return {"error": "No dataset is loaded."}

    if group_column not in df.columns:
        return {
            "error": f"Column '{group_column}' not found in the dataset.",
            "available_columns": list(df.columns),
        }

    if value_column not in df.columns:
        return {
            "error": f"Column '{value_column}' not found in the dataset.",
            "available_columns": list(df.columns),
        }

    if agg_function not in ALLOWED_AGG_FUNCTIONS:
        return {
            "error": f"Aggregation function '{agg_function}' is not supported.",
            "allowed_functions": sorted(ALLOWED_AGG_FUNCTIONS),
        }

    if agg_function != "count" and not pd.api.types.is_numeric_dtype(df[value_column]):
        return {
            "error": (
                f"Column '{value_column}' is not numeric, so '{agg_function}' "
                "cannot be computed on it. Try 'count' instead, or pick a numeric column."
            )
        }

    working_df = df

    if filter_values:
        working_df = df[df[group_column].astype(str).isin([str(v) for v in filter_values])]
        if working_df.empty:
            return {
                "error": f"No rows matched the requested values for '{group_column}'.",
                "requested_values": filter_values,
            }

    try:
        grouped, is_datetime = aggregate_series(working_df, group_column, value_column, agg_function)
    except Exception as e:
        return {"error": f"Groupby operation failed: {e}"}

    result = {}
    for key, value in grouped.items():
        if pd.isna(value):
            result[str(key)] = None
        else:
            try:
                result[str(key)] = round(float(value), 2)
            except (TypeError, ValueError):
                result[str(key)] = str(value)

    note = None
    total_groups = len(result)

    if top_n is not None:
        result = apply_top_n(result, top_n, sort_order or "desc")
    elif total_groups > MAX_GROUPS_RETURNED:
        result = apply_top_n(result, MAX_GROUPS_RETURNED, "desc")
        note = (
            f"Showing the top {MAX_GROUPS_RETURNED} of {total_groups} groups by value. "
            "Ask for a specific top_n or filter_values to see others."
        )

    numeric_entries = [(k, v) for k, v in result.items() if isinstance(v, (int, float))]
    best_group = max(numeric_entries, key=lambda kv: kv[1])[0] if numeric_entries else None
    worst_group = min(numeric_entries, key=lambda kv: kv[1])[0] if numeric_entries else None

    ranking = [k for k, _ in sorted(numeric_entries, key=lambda kv: kv[1], reverse=True)]

    output = {
        "group_column": group_column,
        "value_column": value_column,
        "agg_function": agg_function,
        "group_column_type": "datetime" if is_datetime else str(df[group_column].dtype),
        "result": result,
        "best_group": best_group,
        "worst_group": worst_group,
        "ranking": ranking,
    }

    if len(numeric_entries) == 2:
        (name1, val1), (name2, val2) = numeric_entries[0], numeric_entries[1]
        abs_diff = round(abs(val2 - val1), 2)
        mean_val = (val1 + val2) / 2.0
        pct_diff = round(100.0 * abs_diff / mean_val, 2) if mean_val != 0 else None
        pct_change = round(100.0 * (val2 - val1) / val1, 2) if val1 != 0 else None

        output["comparison"] = {
            "group_1": name1,
            "value_1": val1,
            "group_2": name2,
            "value_2": val2,
            "absolute_difference": abs_diff,
            "percentage_difference": pct_diff,
            "percentage_change": pct_change,
        }

    if filter_values:
        output["filter_applied"] = {"column": group_column, "values": filter_values}

    if note:
        output["note"] = note

    return output


GROUPBY_ANALYSIS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "groupby_analysis",
        "description": (
            "Use this tool whenever the user asks for a calculation BY, PER, "
            "FOR EACH, or ACROSS categories or groups (not time periods -- "
            "use time_analysis for day/week/month/quarter/year breakdowns). "
            "Examples: 'average salary by department', 'total sales by store', "
            "'which store has the highest average sales', "
            "'top 10 stores by average sales', 'bottom 5 regions by revenue', "
            "'compare stores A, B and C'. "
            "Set top_n + sort_order for 'top N' / 'bottom N' questions. "
            "Set filter_values to restrict the analysis to specific category "
            "values mentioned earlier in the conversation (e.g. 'compare "
            "those stores')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_column": {
                    "type": "string",
                    "description": "The categorical column defining the groups, e.g. 'department' or 'Store'.",
                },
                "value_column": {
                    "type": "string",
                    "description": "The numeric column to aggregate, e.g. 'salary' or 'Weekly_Sales'.",
                },
                "agg_function": {
                    "type": "string",
                    "enum": sorted(ALLOWED_AGG_FUNCTIONS),
                    "description": "Aggregation to apply. Defaults to 'mean'.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "If set, only return the top N groups by value (use with sort_order).",
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "'desc' for highest first (top N), 'asc' for lowest first (bottom N).",
                },
                "filter_values": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict group_column to only these values, e.g. specific store names.",
                },
            },
            "required": ["group_column", "value_column"],
        },
    },
}