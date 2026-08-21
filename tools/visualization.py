from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tools.groupby import ALLOWED_AGG_FUNCTIONS, MAX_GROUPS_RETURNED, aggregate_series, apply_top_n
from tools.date_utils import ALLOWED_PERIODS, add_period_column


ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "histogram", "box"}

_AGG_LABELS = {
    "mean": "Average", "sum": "Total", "count": "Count of",
    "min": "Minimum", "max": "Maximum", "median": "Median", "std": "Std Dev of",
}


def create_visualization(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str,
    y_column: str | None = None,
    agg_function: str | None = None,
    period: str | None = None,
    filter_values: list[str | int | float] | None = None,
    top_n: int | None = None,
    sort_order: str | None = None,
) -> dict[str, Any]:
    """
    Create a Plotly visualization. When agg_function is given, the tool
    aggregates first (grouping by category, or by time period for a
    datetime x_column) instead of plotting every raw row -- this is what
    makes "show average sales by store" plot the average, not raw rows.
    """
    if df is None:
        return {"error": "No dataset is loaded."}

    if chart_type not in ALLOWED_CHART_TYPES:
        return {
            "error": f"Chart type '{chart_type}' is not supported.",
            "allowed_chart_types": sorted(ALLOWED_CHART_TYPES),
        }

    if x_column not in df.columns:
        return {
            "error": f"Column '{x_column}' not found in the dataset.",
            "available_columns": list(df.columns),
        }

    if y_column is not None and y_column not in df.columns:
        return {
            "error": f"Column '{y_column}' not found in the dataset.",
            "available_columns": list(df.columns),
        }

    # NOTE: 'box' now requires y_column too -- the chart-building code
    # always referenced y_column in the title, so a box request without
    # one would previously have produced a broken/misleading chart.
    if chart_type in {"bar", "line", "scatter", "box"} and y_column is None:
        return {"error": f"Chart type '{chart_type}' requires a y_column."}

    if agg_function is not None:
        if agg_function not in ALLOWED_AGG_FUNCTIONS:
            return {
                "error": f"Aggregation function '{agg_function}' is not supported.",
                "allowed_functions": sorted(ALLOWED_AGG_FUNCTIONS),
            }
        if chart_type == "histogram":
            return {"error": "Aggregation is not applicable to histogram charts. Remove agg_function or choose a different chart type."}
        if y_column is None:
            return {"error": "agg_function requires a y_column to aggregate."}

    working_df = df.copy()

    if filter_values:
        working_df = working_df[working_df[x_column].astype(str).isin([str(v) for v in filter_values])]
        if working_df.empty:
            return {
                "error": f"No rows matched the requested values for '{x_column}'.",
                "requested_values": filter_values,
            }

    x_is_datetime = pd.api.types.is_datetime64_any_dtype(df[x_column])
    y_is_numeric = y_column is not None and pd.api.types.is_numeric_dtype(df[y_column])
    aggregated = agg_function is not None
    used_period = None

    try:
        if aggregated:
            working_df = working_df.dropna(subset=[x_column, y_column])

            if x_is_datetime:
                used_period = period or "month"
                if used_period not in ALLOWED_PERIODS:
                    return {
                        "error": f"Time period '{used_period}' is not supported.",
                        "allowed_periods": sorted(ALLOWED_PERIODS),
                    }

                working_df["_bucket"] = add_period_column(working_df[x_column], used_period)
                grouped, _ = aggregate_series(working_df, "_bucket", y_column, agg_function)
                chart_df = pd.DataFrame({x_column: grouped.index, y_column: grouped.values})

            else:
                grouped, _ = aggregate_series(working_df, x_column, y_column, agg_function)

                result_dict: dict[str, Any] = {}
                for key, value in grouped.items():
                    result_dict[str(key)] = None if pd.isna(value) else round(float(value), 2)

                if top_n is not None:
                    result_dict = apply_top_n(result_dict, top_n, sort_order or "desc")
                elif len(result_dict) > MAX_GROUPS_RETURNED:
                    result_dict = apply_top_n(result_dict, MAX_GROUPS_RETURNED, "desc")

                chart_df = pd.DataFrame({
                    x_column: list(result_dict.keys()),
                    y_column: list(result_dict.values()),
                })

        else:
            required_columns = [x_column] + ([y_column] if y_column else [])
            chart_df = working_df.dropna(subset=required_columns)

            if x_is_datetime:
                chart_df = chart_df.sort_values(by=x_column)

    except Exception as e:
        return {"error": f"Failed to prepare chart data: {e}"}

    agg_label = _AGG_LABELS.get(agg_function, "")

    try:
        fig: go.Figure

        if chart_type == "bar":
            title = f"{agg_label} {y_column} by {x_column}".strip() if aggregated else f"{y_column} by {x_column}"
            fig = px.bar(chart_df, x=x_column, y=y_column, title=title)

        elif chart_type == "line":
            title = f"{agg_label} {y_column} over {x_column}".strip() if aggregated else f"{y_column} over {x_column}"
            fig = px.line(chart_df, x=x_column, y=y_column, title=title, markers=True)

        elif chart_type == "scatter":
            fig = px.scatter(chart_df, x=x_column, y=y_column, title=f"{y_column} vs {x_column}")

        elif chart_type == "histogram":
            fig = px.histogram(chart_df, x=x_column, title=f"Distribution of {x_column}")

        elif chart_type == "box":
            fig = px.box(chart_df, x=x_column, y=y_column, title=f"Box plot of {y_column}")

        else:
            return {"error": "Unsupported chart type."}

    except Exception as e:
        return {"error": f"Failed to create chart: {e}"}

    if aggregated:
        description = f"A {chart_type} chart showing the {agg_function} of {y_column} by {x_column}"
        if used_period:
            description += f" (bucketed by {used_period})"
        description += "."
    else:
        description = f"A {chart_type} chart was created using {x_column}" + (f" and {y_column}." if y_column else ".")

    return {
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
        "aggregated": aggregated,
        "agg_function": agg_function,
        "period": used_period,
        "rows_used": int(len(chart_df)),
        "description": description,
        "figure": fig,
    }


CREATE_VISUALIZATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_visualization",
        "description": (
            "Create a chart from the dataset: bar, line, scatter, histogram, or box. "
            "IMPORTANT: if the user wants an aggregated view (e.g. 'average sales by "
            "store', 'total sales by month', 'top 10 stores'), set agg_function -- "
            "otherwise every raw row gets plotted instead of the aggregated values. "
            "For a datetime x_column with agg_function set, also set period "
            "(day/week/month/quarter/year) to control the time bucket -- use a LINE "
            "chart for trends over time. For category comparisons, use BAR. For a "
            "'top N' bar chart, also set top_n and sort_order. Use filter_values to "
            "restrict to specific categories mentioned earlier (e.g. 'those stores'). "
            "For a single numeric column's distribution, use HISTOGRAM. For comparing "
            "distributions across categories, use BOX."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "The name of the dataset to analyze (e.g. 'sales.csv'). Optional. Defaults to the primary dataset."},
                "chart_type": {
                    "type": "string",
                    "enum": sorted(ALLOWED_CHART_TYPES),
                    "description": "bar for comparisons, line for trends/time series, scatter for relationships, histogram for distributions, box for distribution comparisons.",
                },
                "x_column": {
                    "type": "string",
                    "description": "Column for the x-axis (or the single column for a histogram).",
                },
                "y_column": {
                    "type": "string",
                    "description": "Numeric column for the y-axis. Not required for histograms.",
                },
                "agg_function": {
                    "type": "string",
                    "enum": sorted(ALLOWED_AGG_FUNCTIONS),
                    "description": "Set this to aggregate before plotting instead of plotting raw rows.",
                },
                "period": {
                    "type": "string",
                    "enum": sorted(ALLOWED_PERIODS),
                    "description": "Time bucket size when x_column is a date and agg_function is set. Defaults to 'month'.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "For aggregated bar charts: only plot the top N categories.",
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "'desc' for highest first, 'asc' for lowest first. Used with top_n.",
                },
                "filter_values": {
                    "type": "array",
                    "items": {"type": ["string", "number"]},
                    "description": "Restrict x_column to exact string or numeric category values from earlier in the conversation.",
                },
            },
            "required": ["chart_type", "x_column"],
        },
    },
}


MULTI_DATASET_VISUALIZATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_multi_dataset_visualization",
        "description": (
            "Create a multi-series comparison chart (bar or line) combining data across multiple datasets "
            "on a common x-axis without requiring an explicit dataset join."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar"],
                    "description": "Type of multi-dataset chart: 'line' or 'bar'. Defaults to 'line'.",
                },
                "series": {
                    "type": "array",
                    "description": "List of series configurations to plot on the same figure.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dataset_name": {
                                "type": "string",
                                "description": "The dataset containing this series.",
                            },
                            "x_column": {
                                "type": "string",
                                "description": "Column for x-axis.",
                            },
                            "y_column": {
                                "type": "string",
                                "description": "Numeric column for y-axis.",
                            },
                            "name": {
                                "type": "string",
                                "description": "Display label for this series (optional).",
                            },
                        },
                        "required": ["dataset_name", "x_column", "y_column"],
                    },
                },
                "title": {
                    "type": "string",
                    "description": "Custom chart title (optional).",
                },
            },
            "required": ["series"],
        },
    },
}


def create_multi_dataset_visualization(
    datasets: dict[str, pd.DataFrame],
    series: list[dict[str, Any]],
    chart_type: str = "line",
    title: str | None = None,
) -> dict[str, Any]:
    """
    Generate a Plotly multi-series visualization comparing traces from different datasets.
    """
    if not datasets:
        return {"error": "No datasets available for multi-dataset visualization."}

    if chart_type not in ["line", "bar"]:
        return {"error": f"Chart type '{chart_type}' is not supported for multi-dataset visualization. Choose 'line' or 'bar'."}

    if not series or not isinstance(series, list) or len(series) == 0:
        return {"error": "At least one series specification is required in 'series'."}

    fig = go.Figure()
    traces_added = 0

    for idx, s_spec in enumerate(series):
        ds_name = s_spec.get("dataset_name")
        x_col = s_spec.get("x_column")
        y_col = s_spec.get("y_column")
        trace_label = s_spec.get("name") or f"{ds_name}: {y_col}"

        if not ds_name or ds_name not in datasets:
            return {
                "error": f"Dataset '{ds_name}' specified in series not found.",
                "available_datasets": list(datasets.keys()),
            }

        df = datasets[ds_name]

        if x_col not in df.columns:
            return {
                "error": f"Column '{x_col}' not found in dataset '{ds_name}'.",
                "available_columns": list(df.columns),
            }

        if y_col not in df.columns:
            return {
                "error": f"Column '{y_col}' not found in dataset '{ds_name}'.",
                "available_columns": list(df.columns),
            }

        if not pd.api.types.is_numeric_dtype(df[y_col]):
            return {"error": f"Column '{y_col}' in dataset '{ds_name}' is not numeric."}

        # Robustly extract x and y series and drop rows with NaNs in either
        try:
            s_x = df[x_col]
            s_y = df[y_col]
        except Exception:
            # Column existence was already checked above, but guard defensively
            return {"error": f"Failed to access columns '{x_col}' or '{y_col}' in dataset '{ds_name}'."}

        # When columns have identical names (x_col == y_col) or duplicated labels,
        # ensure we operate on Series objects and align masks by index.
        if isinstance(s_x, pd.DataFrame):
            # multiple columns with same label -> take first
            s_x = s_x.iloc[:, 0]
        if isinstance(s_y, pd.DataFrame):
            s_y = s_y.iloc[:, 0]

        mask = s_x.notna() & s_y.notna()
        x_vals = s_x.loc[mask].tolist()
        y_vals = s_y.loc[mask].tolist()

        if not x_vals or not y_vals:
            return {"error": f"No data available after dropping nulls for series in dataset '{ds_name}'."}

        if chart_type == "bar":
            fig.add_trace(go.Bar(
                x=x_vals,
                y=y_vals,
                name=trace_label,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=trace_label,
            ))
        traces_added += 1

    chart_title = title or f"Cross-Dataset Comparison ({chart_type.capitalize()})"
    fig.update_layout(
        title=chart_title,
        template="plotly_white",
        legend_title="Series",
    )

    return {
        "status": "success",
        "chart_type": chart_type,
        "traces_count": traces_added,
        "title": chart_title,
        "figure": fig,
    }
