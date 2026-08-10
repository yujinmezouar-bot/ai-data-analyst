from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


ALLOWED_CHART_TYPES = {
    "bar",
    "line",
    "scatter",
    "histogram",
    "box",
}


ALLOWED_DATE_GRANULARITIES = {
    "day",
    "week",
    "month",
    "quarter",
    "year",
}


def _create_date_group(
    series: pd.Series,
    granularity: str,
) -> pd.Series:
    """
    Convert a datetime Series into the requested
    time granularity.
    """

    if granularity == "day":

        return series.dt.floor("D")

    if granularity == "week":

        return (
            series
            - pd.to_timedelta(
                series.dt.weekday,
                unit="D",
            )
        ).dt.floor("D")

    if granularity == "month":

        return (
            series
            .dt.to_period("M")
            .dt.to_timestamp()
        )

    if granularity == "quarter":

        return (
            series
            .dt.to_period("Q")
            .dt.to_timestamp()
        )

    if granularity == "year":

        return (
            series
            .dt.to_period("Y")
            .dt.to_timestamp()
        )

    raise ValueError(
        f"Unsupported date granularity: {granularity}"
    )


def create_visualization(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str,
    y_column: str | None = None,
    date_granularity: str | None = None,
) -> dict[str, Any]:
    """
    Build a Plotly chart from the dataset.

    Supports normal charts and time-based charts.

    Examples:

        bar:
            x=department
            y=salary

        line:
            x=Date
            y=Weekly_Sales
            date_granularity=month

    Returns:

        {
            "figure": Plotly Figure,
            ...
        }
    """

    if df is None:
        return {
            "error": "No dataset is loaded."
        }

    # ---------------------------------------------------------
    # Validate chart type
    # ---------------------------------------------------------

    if chart_type not in ALLOWED_CHART_TYPES:
        return {
            "error": (
                f"Chart type '{chart_type}' "
                "is not supported."
            ),
            "allowed_chart_types": sorted(
                ALLOWED_CHART_TYPES
            ),
        }

    # ---------------------------------------------------------
    # Validate x column
    # ---------------------------------------------------------

    if x_column not in df.columns:
        return {
            "error": (
                f"Column '{x_column}' "
                "not found in the dataset."
            ),
            "available_columns": list(df.columns),
        }

    # ---------------------------------------------------------
    # Validate y column
    # ---------------------------------------------------------

    if (
        y_column is not None
        and y_column not in df.columns
    ):
        return {
            "error": (
                f"Column '{y_column}' "
                "not found in the dataset."
            ),
            "available_columns": list(df.columns),
        }

    # ---------------------------------------------------------
    # Validate y column requirement
    # ---------------------------------------------------------

    if (
        chart_type in {
            "bar",
            "line",
            "scatter",
        }
        and y_column is None
    ):
        return {
            "error": (
                f"Chart type '{chart_type}' "
                "requires a y_column."
            )
        }

    # ---------------------------------------------------------
    # Validate date granularity
    # ---------------------------------------------------------

    if date_granularity is not None:

        if (
            date_granularity
            not in ALLOWED_DATE_GRANULARITIES
        ):
            return {
                "error": (
                    f"Date granularity "
                    f"'{date_granularity}' "
                    "is not supported."
                ),
                "allowed_date_granularities": sorted(
                    ALLOWED_DATE_GRANULARITIES
                ),
            }

        if not pd.api.types.is_datetime64_any_dtype(
            df[x_column]
        ):
            return {
                "error": (
                    f"Column '{x_column}' is not "
                    "recognized as a datetime column. "
                    "Make sure date detection is working "
                    "correctly."
                )
            }

    try:

        # -----------------------------------------------------
        # Prepare data
        # -----------------------------------------------------

        plot_df = df.copy()

        actual_x_column = x_column

        # -----------------------------------------------------
        # Date-based visualization
        # -----------------------------------------------------

        if date_granularity is not None:

            plot_df["_date_group"] = _create_date_group(
                plot_df[x_column],
                date_granularity,
            )

            actual_x_column = "_date_group"

            # Sort chronologically.
            plot_df = plot_df.sort_values(
                "_date_group"
            )

        # -----------------------------------------------------
        # Create chart
        # -----------------------------------------------------

        if chart_type == "bar":

            # For date-based bar charts, aggregate the
            # y values so that "sales by month" means
            # actual monthly sales rather than one bar
            # per original row.

            if date_granularity is not None:

                plot_df = (
                    plot_df
                    .groupby(
                        actual_x_column,
                        dropna=False,
                    )[y_column]
                    .sum()
                    .reset_index()
                )

            fig = px.bar(
                plot_df,
                x=actual_x_column,
                y=y_column,
                title=(
                    f"{y_column} by "
                    f"{date_granularity or x_column}"
                ),
            )

        elif chart_type == "line":

            if date_granularity is not None:

                plot_df = (
                    plot_df
                    .groupby(
                        actual_x_column,
                        dropna=False,
                    )[y_column]
                    .sum()
                    .reset_index()
                )

            fig = px.line(
                plot_df,
                x=actual_x_column,
                y=y_column,
                title=(
                    f"{y_column} over "
                    f"{date_granularity or x_column}"
                ),
                markers=True,
            )

        elif chart_type == "scatter":

            fig = px.scatter(
                plot_df,
                x=actual_x_column,
                y=y_column,
                title=(
                    f"{y_column} vs "
                    f"{x_column}"
                ),
            )

        elif chart_type == "histogram":

            fig = px.histogram(
                plot_df,
                x=actual_x_column,
                title=(
                    f"Distribution of "
                    f"{x_column}"
                ),
            )

        elif chart_type == "box":

            fig = px.box(
                plot_df,
                x=actual_x_column,
                y=y_column,
                title=(
                    f"Box plot of "
                    f"{y_column}"
                ),
            )

        else:
            return {
                "error": (
                    f"Unsupported chart type: "
                    f"{chart_type}"
                )
            }

    except Exception as e:

        return {
            "error": (
                f"Failed to create chart: {e}"
            )
        }

    # ---------------------------------------------------------
    # Return metadata + figure
    # ---------------------------------------------------------

    return {
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
        "date_granularity": date_granularity,
        "description": (
            f"A {chart_type} chart was created "
            f"using {x_column}"
            + (
                f" grouped by {date_granularity}"
                if date_granularity
                else ""
            )
            + (
                f" and {y_column}."
                if y_column
                else "."
            )
        ),
        "figure": fig,
    }


CREATE_VISUALIZATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_visualization",
        "description": (
            "Create a bar, line, scatter, histogram, "
            "or box plot from the dataset. "
            ""
            "Use this when the user asks to see, plot, "
            "chart, or visualize data. "
            ""
            "Examples: "
            "'show me a bar chart of sales by region', "
            "'plot the distribution of ages', "
            "'show monthly sales', "
            "'plot sales by year', "
            "'show weekly sales over time'. "
            ""
            "For time-based questions, x_column should "
            "be the detected datetime column and "
            "date_granularity should be one of: "
            "day, week, month, quarter, or year. "
            ""
            "For example, for 'show monthly sales', use "
            "chart_type='line', "
            "x_column='Date', "
            "y_column='Sales', "
            "date_granularity='month'."
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
                        "The type of chart to create."
                    ),
                },

                "x_column": {
                    "type": "string",
                    "description": (
                        "The column used for the "
                        "x-axis. For time-based "
                        "charts this should be the "
                        "datetime column."
                    ),
                },

                "y_column": {
                    "type": "string",
                    "description": (
                        "The numeric column used "
                        "for the y-axis."
                    ),
                },

                "date_granularity": {
                    "type": "string",
                    "enum": sorted(
                        ALLOWED_DATE_GRANULARITIES
                    ),
                    "description": (
                        "Use only for datetime columns. "
                        "Choose day, week, month, "
                        "quarter, or year. "
                        "For example, use 'month' "
                        "for 'monthly sales'."
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
