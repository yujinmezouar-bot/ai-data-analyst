from typing import Any

import pandas as pd


# ============================================================
# ALLOWED AGGREGATIONS
# ============================================================

ALLOWED_AGG_FUNCTIONS = {
    "mean",
    "sum",
    "count",
    "min",
    "max",
    "median",
    "std",
}


# ============================================================
# GROUPBY ANALYSIS
# ============================================================

def groupby_analysis(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    agg_function: str = "mean",
) -> dict[str, Any]:
    """
    Group the dataset by group_column and aggregate
    value_column.

    Supports normal categorical columns as well as
    pandas datetime columns.
    """

    if df is None:

        return {
            "error": "No dataset is loaded."
        }


    # --------------------------------------------------------
    # Check group column
    # --------------------------------------------------------

    if group_column not in df.columns:

        return {
            "error": (
                f"Column '{group_column}' "
                "not found in the dataset."
            ),
            "available_columns": list(
                df.columns
            ),
        }


    # --------------------------------------------------------
    # Check value column
    # --------------------------------------------------------

    if value_column not in df.columns:

        return {
            "error": (
                f"Column '{value_column}' "
                "not found in the dataset."
            ),
            "available_columns": list(
                df.columns
            ),
        }


    # --------------------------------------------------------
    # Check aggregation
    # --------------------------------------------------------

    if (
        agg_function
        not in ALLOWED_AGG_FUNCTIONS
    ):

        return {
            "error": (
                f"Aggregation function "
                f"'{agg_function}' is not supported."
            ),
            "allowed_functions": sorted(
                ALLOWED_AGG_FUNCTIONS
            ),
        }


    # --------------------------------------------------------
    # Check numeric value column
    # --------------------------------------------------------

    if (
        agg_function != "count"
        and not pd.api.types.is_numeric_dtype(
            df[value_column]
        )
    ):

        return {
            "error": (
                f"Column '{value_column}' "
                f"is not numeric, so "
                f"'{agg_function}' cannot be "
                "computed on it. "
                "Try 'count' instead, or pick "
                "a numeric column."
            )
        }


    try:

        working_df = df.copy()


        # ----------------------------------------------------
        # Datetime grouping
        # ----------------------------------------------------

        is_datetime = (
            pd.api.types.is_datetime64_any_dtype(
                working_df[group_column]
            )
        )


        if is_datetime:

            # Remove missing dates

            working_df = working_df.dropna(
                subset=[group_column]
            )


            grouped = (
                working_df
                .groupby(group_column)[
                    value_column
                ]
                .agg(agg_function)
                .sort_index()
            )


        else:

            grouped = (
                working_df
                .groupby(group_column)[
                    value_column
                ]
                .agg(agg_function)
            )


    except Exception as e:

        return {
            "error": (
                f"Groupby operation failed: {e}"
            )
        }


    # --------------------------------------------------------
    # Convert result to JSON-safe values
    # --------------------------------------------------------

    result = {}

    for key, value in grouped.items():

        if pd.isna(value):

            result[str(key)] = None

        else:

            try:

                result[str(key)] = round(
                    float(value),
                    2,
                )

            except (TypeError, ValueError):

                result[str(key)] = str(value)


    return {
        "group_column": group_column,
        "value_column": value_column,
        "agg_function": agg_function,
        "group_column_type": (
            "datetime"
            if is_datetime
            else str(
                df[group_column].dtype
            )
        ),
        "result": result,
    }


# ============================================================
# TOOL SCHEMA
# ============================================================

GROUPBY_ANALYSIS_SCHEMA = {
    "type": "function",

    "function": {

        "name": "groupby_analysis",

        "description": (
            "Use this tool whenever the user asks "
            "for a calculation BY, PER, FOR EACH, "
            "or ACROSS categories, groups, dates, "
            "months, or time periods. "

            "Examples include: "
            "'average salary by department', "
            "'average weekly sales by store', "
            "'total sales per region', "
            "'median salary by department', "
            "'minimum sales for each store', "
            "'maximum sales per region', "
            "'average sales by date', "
            "'total sales by month', "
            "or 'number of orders per customer'. "

            "group_column is the column defining "
            "the groups. It can be categorical "
            "or datetime. "

            "value_column is the column to aggregate. "

            "agg_function specifies the calculation "
            "such as mean, sum, count, min, max, "
            "median, or std."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "group_column": {

                    "type": "string",

                    "description": (
                        "The column defining the groups. "
                        "This may be a categorical column "
                        "such as 'department' or 'region', "
                        "or a datetime column such as "
                        "'Date'."
                    ),
                },

                "value_column": {

                    "type": "string",

                    "description": (
                        "The numeric column to aggregate, "
                        "such as 'salary', 'sales', "
                        "or 'amount'."
                    ),
                },

                "agg_function": {

                    "type": "string",

                    "enum": sorted(
                        ALLOWED_AGG_FUNCTIONS
                    ),

                    "description": (
                        "The aggregation function to apply. "
                        "Use mean for average, sum for total, "
                        "count for number of records, min "
                        "for minimum, max for maximum, "
                        "median for median, and std for "
                        "standard deviation."
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
