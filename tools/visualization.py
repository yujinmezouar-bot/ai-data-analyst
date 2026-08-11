from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# ALLOWED CHART TYPES
# ============================================================

ALLOWED_CHART_TYPES = {
    "bar",
    "line",
    "scatter",
    "histogram",
    "box",
}


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str,
    y_column: str | None = None,
) -> dict[str, Any]:
    """
    Create a Plotly visualization.

    Supports categorical, numeric and datetime columns.

    Datetime columns are automatically sorted chronologically
    before creating line/scatter charts.
    """

    if df is None:

        return {
            "error": "No dataset is loaded."
        }


    # --------------------------------------------------------
    # Validate chart type
    # --------------------------------------------------------

    if (
        chart_type
        not in ALLOWED_CHART_TYPES
    ):

        return {
            "error": (
                f"Chart type '{chart_type}' "
                "is not supported."
            ),
            "allowed_chart_types": sorted(
                ALLOWED_CHART_TYPES
            ),
        }


    # --------------------------------------------------------
    # Validate X column
    # --------------------------------------------------------

    if x_column not in df.columns:

        return {
            "error": (
                f"Column '{x_column}' "
                "not found in the dataset."
            ),
            "available_columns": list(
                df.columns
            ),
        }


    # --------------------------------------------------------
    # Validate Y column
    # --------------------------------------------------------

    if (
        y_column is not None
        and y_column not in df.columns
    ):

        return {
            "error": (
                f"Column '{y_column}' "
                "not found in the dataset."
            ),
            "available_columns": list(
                df.columns
            ),
        }


    # --------------------------------------------------------
    # Validate chart requirements
    # --------------------------------------------------------

    if (
        chart_type
        in {"bar", "line", "scatter"}
        and y_column is None
    ):

        return {
            "error": (
                f"Chart type '{chart_type}' "
                "requires a y_column."
            )
        }


    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    working_df = df.copy()


    x_is_datetime = (
        pd.api.types.is_datetime64_any_dtype(
            working_df[x_column]
        )
    )


    y_is_numeric = (
        y_column is not None
        and pd.api.types.is_numeric_dtype(
            working_df[y_column]
        )
    )


    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    required_columns = [x_column]

    if y_column is not None:

        required_columns.append(
            y_column
        )


    working_df = working_df.dropna(
        subset=required_columns
    )


    # --------------------------------------------------------
    # Sort datetime columns
    # --------------------------------------------------------

    if x_is_datetime:

        working_df = working_df.sort_values(
            by=x_column
        )


    # --------------------------------------------------------
    # Create chart
    # --------------------------------------------------------

    try:

        fig: go.Figure


        # ====================================================
        # BAR
        # ====================================================

        if chart_type == "bar":

            fig = px.bar(
                working_df,
                x=x_column,
                y=y_column,
                title=(
                    f"{y_column} by "
                    f"{x_column}"
                ),
            )


        # ====================================================
        # LINE
        # ====================================================

        elif chart_type == "line":

            fig = px.line(
                working_df,
                x=x_column,
                y=y_column,
                title=(
                    f"{y_column} over "
                    f"{x_column}"
                ),
                markers=True,
            )


        # ====================================================
        # SCATTER
        # ====================================================

        elif chart_type == "scatter":

            fig = px.scatter(
                working_df,
                x=x_column,
                y=y_column,
                title=(
                    f"{y_column} vs "
                    f"{x_column}"
                ),
            )


        # ====================================================
        # HISTOGRAM
        # ====================================================

        elif chart_type == "histogram":

            fig = px.histogram(
                working_df,
                x=x_column,
                title=(
                    f"Distribution of "
                    f"{x_column}"
                ),
            )


        # ====================================================
        # BOX
        # ====================================================

        elif chart_type == "box":

            fig = px.box(
                working_df,
                x=x_column,
                y=y_column,
                title=(
                    f"Box plot of "
                    f"{y_column}"
                ),
            )


        # ====================================================
        # UNKNOWN
        # ====================================================

        else:

            return {
                "error": (
                    "Unsupported chart type."
                )
            }


    except Exception as e:

        return {
            "error": (
                f"Failed to create chart: {e}"
            )
        }


    # --------------------------------------------------------
    # Add useful metadata
    # --------------------------------------------------------

    x_type = str(
        working_df[x_column].dtype
    )


    y_type = (
        str(
            working_df[y_column].dtype
        )
        if y_column is not None
        else None
    )


    return {
        "chart_type": chart_type,

        "x_column": x_column,

        "y_column": y_column,

        "x_column_type": (
            "datetime"
            if x_is_datetime
            else x_type
        ),

        "y_column_type": (
            "numeric"
            if y_is_numeric
            else y_type
        ),

        "rows_used": int(
            len(working_df)
        ),

        "description": (
            f"A {chart_type} chart was created "
            f"using {x_column}"
            + (
                f" and {y_column}."
                if y_column
                else "."
            )
        ),

        "figure": fig,
    }


# ============================================================
# TOOL SCHEMA
# ============================================================

CREATE_VISUALIZATION_SCHEMA = {

    "type": "function",

    "function": {

        "name": "create_visualization",

        "description": (
            "Create a chart from the dataset. "

            "Use this tool whenever the user asks "
            "to show, draw, plot, visualize, graph, "
            "or chart data. "

            "Supported charts are bar, line, scatter, "
            "histogram, and box. "

            "IMPORTANT: When the user asks for a trend "
            "over time, sales over time, data by date, "
            "data by month, or a chronological trend, "
            "use a LINE chart and select the datetime "
            "column as x_column. "

            "For comparisons between categories, "
            "use a BAR chart. "

            "For distribution of one numeric column, "
            "use a HISTOGRAM. "

            "For comparing distributions between "
            "categories, use a BOX plot. "

            "For bar, line, scatter and box charts, "
            "y_column is normally required. "

            "For histogram, only x_column is needed."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "chart_type": {

                    "type": "string",

                    "enum": sorted(
                        ALLOWED_CHART_TYPES
                    ),

                    "description": (
                        "Chart type: "
                        "bar for category comparison, "
                        "line for trends/time series, "
                        "scatter for relationships, "
                        "histogram for distributions, "
                        "box for distribution comparison."
                    ),
                },

                "x_column": {

                    "type": "string",

                    "description": (
                        "Column used on the x-axis. "
                        "For time-series questions, "
                        "this should be the datetime "
                        "column."
                    ),
                },

                "y_column": {

                    "type": "string",

                    "description": (
                        "Numeric column used on the "
                        "y-axis. Not required for "
                        "histograms."
                    ),
                },
            },

            "required": [
                "chart_type",
                "x_column",
            ],
        },
    },
}
