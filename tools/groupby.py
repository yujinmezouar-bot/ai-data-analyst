from typing import Any

import pandas as pd


# Aggregation functions allowed for the LLM.
# Explicit whitelist for safety.
ALLOWED_AGG_FUNCTIONS = {
    "mean",
    "sum",
    "count",
    "min",
    "max",
    "median",
    "std",
}


# Date granularities supported by this tool.
ALLOWED_DATE_GRANULARITIES = {
    "day",
    "week",
    "month",
    "quarter",
    "year",
}


def groupby_analysis(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    agg_function: str = "mean",
    date_granularity: str | None = None,
) -> dict[str, Any]:
    """
    Group the dataset by a column and aggregate a value column.

    Supports normal categorical grouping:

        department → salary → mean

    and date grouping:

        Date → Weekly_Sales → sum → month

    Example:

        groupby_analysis(
            df,
            group_column="Date",
            value_column="Weekly_Sales",
            agg_function="sum",
            date_granularity="month",
        )
    """

    if df is None:
        return {
            "error": "No dataset is loaded."
        }

    # ---------------------------------------------------------
    # Validate group column
    # ---------------------------------------------------------

    if group_column not in df.columns:
        return {
            "error": (
                f"Column '{group_column}' "
                "not found in the dataset."
            ),
            "available_columns": list(df.columns),
        }

    # ---------------------------------------------------------
    # Validate value column
    # ---------------------------------------------------------

    if value_column not in df.columns:
        return {
            "error": (
                f"Column '{value_column}' "
                "not found in the dataset."
            ),
            "available_columns": list(df.columns),
        }

    # ---------------------------------------------------------
    # Validate aggregation
    # ---------------------------------------------------------

    if agg_function not in ALLOWED_AGG_FUNCTIONS:
        return {
            "error": (
                f"Aggregation function "
                f"'{agg_function}' is not supported."
            ),
            "allowed_functions": sorted(
                ALLOWED_AGG_FUNCTIONS
            ),
        }

    # ---------------------------------------------------------
    # Validate date granularity
    # ---------------------------------------------------------

    if date_granularity is not None:

        if date_granularity not in ALLOWED_DATE_GRANULARITIES:
            return {
                "error": (
                    f"Date granularity "
                    f"'{date_granularity}' is not supported."
                ),
                "allowed_date_granularities": sorted(
                    ALLOWED_DATE_GRANULARITIES
                ),
            }

        # Date granularity only makes sense with a datetime
        # group column.
        if not pd.api.types.is_datetime64_any_dtype(
            df[group_column]
        ):
            return {
                "error": (
                    f"Column '{group_column}' is not recognized "
                    "as a datetime column. "
                    "Make sure the date column was detected "
                    "correctly when the dataset was loaded."
                )
            }

    # ---------------------------------------------------------
    # Validate value column
    # ---------------------------------------------------------

    if (
        agg_function != "count"
        and not pd.api.types.is_numeric_dtype(
            df[value_column]
        )
    ):
        return {
            "error": (
                f"Column '{value_column}' is not numeric, "
                f"so '{agg_function}' cannot be computed "
                "on it. Try 'count' instead, or choose "
                "a numeric column."
            )
        }

    try:

        working_df = df.copy()

        # -----------------------------------------------------
        # Normal groupby
        # -----------------------------------------------------

        if date_granularity is None:

            grouped = (
                working_df
                .groupby(
                    group_column,
                    dropna=False,
                )[value_column]
                .agg(agg_function)
            )

            result = {}

            for key, value in grouped.items():

                if pd.isna(value):
                    result[str(key)] = None
                else:
                    result[str(key)] = round(
                        float(value),
                        2,
                    )

        # -----------------------------------------------------
        # Date groupby
        # -----------------------------------------------------

        else:

            date_series = working_df[group_column]

            if date_granularity == "day":

                working_df["_date_group"] = (
                    date_series.dt.floor("D")
                )

            elif date_granularity == "week":

                # Start of the week = Monday.
                working_df["_date_group"] = (
                    date_series
                    - pd.to_timedelta(
                        date_series.dt.weekday,
                        unit="D",
                    )
                ).dt.floor("D")

            elif date_granularity == "month":

                working_df["_date_group"] = (
                    date_series.dt.to_period("M")
                    .dt.to_timestamp()
                )

            elif date_granularity == "quarter":

                working_df["_date_group"] = (
                    date_series.dt.to_period("Q")
                    .dt.to_timestamp()
                )

            elif date_granularity == "year":

                working_df["_date_group"] = (
                    date_series.dt.to_period("Y")
                    .dt.to_timestamp()
                )

            grouped = (
                working_df
                .groupby(
                    "_date_group",
                    dropna=False,
                )[value_column]
                .agg(agg_function)
            )

            result = {}

            for key, value in grouped.items():

                if pd.isna(key):

                    key_string = "Unknown"

                else:

                    if date_granularity == "day":
                        key_string = key.strftime(
                            "%Y-%m-%d"
                        )

                    elif date_granularity == "week":
                        key_string = (
                            "Week of "
                            + key.strftime("%Y-%m-%d")
                        )

                    elif date_granularity == "month":
                        key_string = key.strftime(
                            "%Y-%m"
                        )

                    elif date_granularity == "quarter":
                        key_string = (
                            f"{key.year}-Q"
                            f"{((key.month - 1) // 3) + 1}"
                        )

                    elif date_granularity == "year":
                        key_string = key.strftime(
                            "%Y"
                        )

                if pd.isna(value):
                    result[key_string] = None
                else:
                    result[key_string] = round(
                        float(value),
                        2,
                    )

    except Exception as e:

        return {
            "error": (
                f"Groupby operation failed: {e}"
            )
        }

    return {
        "group_column": group_column,
        "value_column": value_column,
        "agg_function": agg_function,
        "date_granularity": date_granularity,
        "result": result,
    }


GROUPBY_ANALYSIS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "groupby_analysis",
        "description": (
            "Calculate an aggregation BY, PER, FOR EACH, "
            "or ACROSS categories or time periods. "
            ""
            "Use this for questions such as: "
            "'average salary by department', "
            "'total sales per region', "
            "'median salary by department', "
            "'average weekly sales by store', "
            "'sales by month', "
            "'total sales per year', "
            "'average sales by week', "
            "or 'sales for each quarter'. "
            ""
            "group_column is the column defining the groups. "
            "value_column is the numeric column to aggregate. "
            "agg_function specifies the calculation: "
            "mean, sum, count, min, max, median, or std. "
            ""
            "IMPORTANT FOR DATE COLUMNS: "
            "If group_column is a date/datetime column and "
            "the user asks for daily, weekly, monthly, "
            "quarterly, or yearly analysis, provide "
            "date_granularity as 'day', 'week', 'month', "
            "'quarter', or 'year'. "
            ""
            "For example, for 'total sales by month', use "
            "group_column='Date', "
            "value_column='Sales', "
            "agg_function='sum', "
            "date_granularity='month'."
        ),
        "parameters": {
            "type": "object",
            "properties": {

                "group_column": {
                    "type": "string",
                    "description": (
                        "The column defining the groups. "
                        "For time analysis this should be "
                        "the detected datetime column."
                    ),
                },

                "value_column": {
                    "type": "string",
                    "description": (
                        "The numeric column to aggregate."
                    ),
                },

                "agg_function": {
                    "type": "string",
                    "enum": sorted(
                        ALLOWED_AGG_FUNCTIONS
                    ),
                    "description": (
                        "Aggregation to apply: "
                        "mean, sum, count, min, max, "
                        "median, or std."
                    ),
                },

                "date_granularity": {
                    "type": "string",
                    "enum": sorted(
                        ALLOWED_DATE_GRANULARITIES
                    ),
                    "description": (
                        "Only use this when group_column "
                        "is a date column. "
                        "Choose day, week, month, quarter, "
                        "or year. "
                        "Leave it out for normal categorical "
                        "grouping."
                    ),
                },
            },

            "required": [
                "group_column",
                "value_column",
            ],
        },
    },
}
