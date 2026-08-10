from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "histogram", "box"}


def create_visualization(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str,
    y_column: str | None = None,
) -> dict[str, Any]:
    """
    Build a Plotly chart from the dataset.

    Returns a dict containing:
    - "figure": the actual Plotly Figure object (NOT JSON-serializable —
      the Agent must strip this key out before sending the result to the LLM)
    - everything else: plain metadata that IS safe to send to the LLM

    Example: chart_type="bar", x_column="department", y_column="salary"
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

    if chart_type in {"bar", "line", "scatter"} and y_column is None:
        return {
            "error": f"Chart type '{chart_type}' requires a y_column, but none was given.",
        }

    try:
        fig: go.Figure
        if chart_type == "bar":
            fig = px.bar(df, x=x_column, y=y_column, title=f"{y_column} by {x_column}")
        elif chart_type == "line":
            fig = px.line(df, x=x_column, y=y_column, title=f"{y_column} over {x_column}")
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_column, y=y_column, title=f"{y_column} vs {x_column}")
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_column, title=f"Distribution of {x_column}")
        elif chart_type == "box":
            fig = px.box(df, x=x_column, y=y_column, title=f"Box plot of {x_column}")
    except Exception as e:
        return {"error": f"Failed to create chart: {e}"}

    return {
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
        "description": f"A {chart_type} chart was created using column(s): {x_column}"
                        + (f" and {y_column}." if y_column else "."),
        "figure": fig,  # stripped out before this dict is sent to the LLM
    }


CREATE_VISUALIZATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_visualization",
        "description": (
            "Create a chart (bar, line, scatter, histogram, or box plot) from the "
            "dataset. Use this when the user asks to see, plot, chart, or visualize "
            "data, e.g. 'show me a bar chart of sales by region' or 'plot the "
            "distribution of ages'. For histogram, only x_column is needed. For "
            "bar/line/scatter/box, both x_column and y_column are typically needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": sorted(ALLOWED_CHART_TYPES),
                    "description": "The type of chart to create.",
                },
                "x_column": {
                    "type": "string",
                    "description": "The column to use for the x-axis (or the single column for a histogram).",
                },
                "y_column": {
                    "type": "string",
                    "description": "The column to use for the y-axis. Not needed for histograms.",
                },
            },
            "required": ["chart_type", "x_column"],
        },
    },
}