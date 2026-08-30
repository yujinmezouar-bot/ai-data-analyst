from typing import Any

import pandas as pd

from tools.date_utils import (
    _name_looks_like_date,
    _parse_date_series,
    add_period_column,
    detect_date_columns,
    format_period_label,
)


MAX_CATEGORY_SAMPLES = 5
MAX_TEMPORAL_COLUMNS = 3
PERIOD_PROFILE_LIMITS = {
    "year": 8,
    "quarter": 8,
    "month": 12,
    "week": 8,
}
IDENTIFIER_KEYWORDS = (
    "id", "identifier", "code", "key", "uuid", "guid", "ssn", "account", "pk", "fk",
    "customer_id", "order_id", "product_id", "user_id", "store_id", "employee_id",
)


def _build_period_profile(series: pd.Series) -> dict[str, Any] | None:
    """Return bounded, canonical period metadata for one datetime-like Series."""
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
    else:
        parsed, _, _ = _parse_date_series(series)
        if parsed is None:
            return None

    populated = parsed.dropna()
    if populated.empty:
        return None

    labels: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for period, limit in PERIOD_PROFILE_LIMITS.items():
        buckets = add_period_column(populated, period).dropna().drop_duplicates().sort_values()
        counts[f"{period}s"] = int(len(buckets))
        labels[period] = [
            format_period_label(pd.Timestamp(bucket), period)
            for bucket in buckets.iloc[-limit:]
        ]

    return {
        "min_date": pd.Timestamp(populated.min()).strftime("%Y-%m-%d"),
        "max_date": pd.Timestamp(populated.max()).strftime("%Y-%m-%d"),
        "years": labels["year"],
        "recent_quarters": labels["quarter"],
        "recent_months": labels["month"],
        "recent_weeks": labels["week"],
        "period_counts": counts,
    }


def _temporal_profile_columns(
    df: pd.DataFrame,
    datetime_columns: list[str],
    date_column_details: dict[str, dict[str, Any]],
) -> list[str]:
    """Choose detailed temporal columns deterministically using existing metadata."""
    column_order = {str(column): index for index, column in enumerate(df.columns)}
    return sorted(
        datetime_columns,
        key=lambda column: (
            -int(pd.api.types.is_datetime64_any_dtype(df[column])),
            -int(_name_looks_like_date(column)),
            -float(date_column_details.get(column, {}).get("confidence", 0.0)),
            column_order[column],
        ),
    )[:MAX_TEMPORAL_COLUMNS]


def build_dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build a comprehensive, compact, and deterministic profile of the dataset.
    Extracts structural, semantic, cardinality, and quality metadata.
    """
    if df is None:
        return {"error": "No dataset is loaded."}

    num_rows = int(df.shape[0])
    num_columns = int(df.shape[1])

    if num_rows == 0 or num_columns == 0:
        return {
            "num_rows": num_rows,
            "num_columns": num_columns,
            "column_names": list(df.columns),
            "column_types": {},
            "semantic_types": {},
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "boolean_columns": [],
            "constant_columns": [],
            "all_null_columns": [],
            "potential_identifiers": [],
            "candidate_group_columns": [],
            "missing_summary": {},
            "date_columns": [],
            "date_column_details": {},
            "temporal_profile_omitted_count": 0,
            "column_profiles": {},
            "memory_usage_kb": 0.0,
        }

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    date_column_details = detect_date_columns(df)
    date_columns = list(date_column_details.keys())

    semantic_types: dict[str, str] = {}
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    boolean_cols: list[str] = []
    constant_cols: list[str] = []
    all_null_cols: list[str] = []
    potential_identifiers: list[dict[str, Any]] = []
    candidate_group_cols: list[str] = []
    missing_summary: dict[str, dict[str, Any]] = {}
    column_profiles: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = round(100.0 * null_count / num_rows, 2)
        valid_series = series.dropna()
        n_valid = len(valid_series)
        n_unique = int(valid_series.nunique()) if n_valid > 0 else 0

        # Quality flags
        is_all_null = (null_count == num_rows)
        is_constant = (n_unique == 1 and not is_all_null)

        if is_all_null:
            all_null_cols.append(str(col))
        if is_constant:
            constant_cols.append(str(col))
        if null_count > 0:
            missing_summary[str(col)] = {
                "missing_count": null_count,
                "missing_percentage": null_pct,
            }

        # Semantic Type Classification
        if str(col) in date_columns or pd.api.types.is_datetime64_any_dtype(series):
            sem_type = "datetime"
            datetime_cols.append(str(col))
        elif pd.api.types.is_bool_dtype(series):
            sem_type = "boolean"
            boolean_cols.append(str(col))
        elif pd.api.types.is_numeric_dtype(series):
            sem_type = "numeric"
            numeric_cols.append(str(col))
        else:
            sem_type = "categorical"
            categorical_cols.append(str(col))

        semantic_types[str(col)] = sem_type

        # Identifier Detection (Heuristic & Conservative)
        col_normalized = str(col).lower().replace("_", "").replace("-", "").replace(" ", "")
        has_id_name = any(
            col_normalized.startswith(kw) or col_normalized.endswith(kw) or kw in col_normalized
            for kw in IDENTIFIER_KEYWORDS
        )
        uniqueness_ratio = round(float(n_unique) / float(n_valid), 3) if n_valid > 0 else 0.0

        id_status = None
        if num_rows >= 5 and n_valid >= 5 and uniqueness_ratio == 1.0 and has_id_name:
            id_status = "detected"
        elif num_rows >= 10 and n_valid >= 10 and uniqueness_ratio >= 0.95 and has_id_name:
            id_status = "detected"
        elif (
            num_rows >= 10 and n_valid >= 10 and uniqueness_ratio == 1.0
            and sem_type in ("categorical", "numeric")
            and not pd.api.types.is_float_dtype(series)
            and not is_constant
        ):
            id_status = "possible"

        if id_status:
            potential_identifiers.append({
                "column": str(col),
                "status": id_status,
                "uniqueness_ratio": uniqueness_ratio,
                "unique_count": n_unique,
            })

        # Category Samples (Bounded)
        sample_values = []
        if sem_type in ("categorical", "boolean") and n_valid > 0:
            raw_unique = valid_series.astype(str).unique()
            sample_values = [str(val) for val in raw_unique[:MAX_CATEGORY_SAMPLES]]

        # Candidate Group Column Suitability
        if sem_type in ("categorical", "boolean") and 1 < n_unique <= 50 and (id_status != "detected"):
            candidate_group_cols.append(str(col))

        column_profiles[str(col)] = {
            "dtype": str(df[col].dtype),
            "semantic_type": sem_type,
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": n_unique,
            "is_constant": is_constant,
            "is_all_null": is_all_null,
            "is_potential_identifier": id_status is not None,
            "sample_values": sample_values if sample_values else None,
        }

    profiled_temporal_columns = _temporal_profile_columns(
        df, datetime_cols, date_column_details
    )
    for column in profiled_temporal_columns:
        period_profile = _build_period_profile(df[column])
        if period_profile is not None:
            date_column_details.setdefault(column, {})["period_profile"] = period_profile

    return {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "column_names": list(df.columns),
        "column_types": dtypes,
        "semantic_types": semantic_types,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "boolean_columns": boolean_cols,
        "constant_columns": constant_cols,
        "all_null_columns": all_null_cols,
        "potential_identifiers": potential_identifiers,
        "candidate_group_columns": candidate_group_cols,
        "missing_summary": missing_summary,
        "date_columns": date_columns,
        "date_column_details": date_column_details,
        "temporal_profile_omitted_count": max(
            0, len(datetime_cols) - len(profiled_temporal_columns)
        ),
        "column_profiles": column_profiles,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }


def _format_temporal_profile_lines(profile: dict[str, Any]) -> list[str]:
    lines = []
    for column in profile.get("datetime_columns", []):
        period_profile = profile.get("date_column_details", {}).get(column, {}).get("period_profile")
        if not period_profile:
            continue
        counts = period_profile["period_counts"]
        lines.append(
            f"- Temporal profile for {column}: range {period_profile['min_date']} -> "
            f"{period_profile['max_date']}; years {period_profile['years']}; "
            f"recent quarters {period_profile['recent_quarters']}; "
            f"recent months {period_profile['recent_months']}; "
            f"recent weeks {period_profile['recent_weeks']}; period counts "
            f"years={counts['years']}, quarters={counts['quarters']}, "
            f"months={counts['months']}, weeks={counts['weeks']}"
        )
    omitted = profile.get("temporal_profile_omitted_count", 0)
    if omitted:
        lines.append(f"- Temporal detail omitted for {omitted} additional datetime column(s).")
    return lines


def format_dataset_context(df: pd.DataFrame) -> str:
    """
    Format a concise, human-readable profile summary of the dataset
    to inject directly into the Agent's working context.
    """
    profile = build_dataset_profile(df)
    if "error" in profile or profile.get("num_rows", 0) == 0:
        return ""

    lines = [
        "[Active Dataset Context]",
        f"Shape: {profile['num_rows']} rows, {profile['num_columns']} columns",
    ]

    if profile["numeric_columns"]:
        lines.append(f"- Numeric columns: {', '.join(profile['numeric_columns'])}")

    if profile["datetime_columns"]:
        date_info = []
        for col in profile["datetime_columns"]:
            fmt = profile["date_column_details"].get(col, {}).get("format", "datetime")
            date_info.append(f"{col} (format: {fmt})")
        lines.append(f"- Datetime columns: {', '.join(date_info)}")
        lines.extend(_format_temporal_profile_lines(profile))

    if profile["categorical_columns"]:
        cat_info = []
        for col in profile["categorical_columns"]:
            samples = profile["column_profiles"].get(col, {}).get("sample_values", [])
            sample_str = f", e.g. {samples}" if samples else ""
            u_count = profile["column_profiles"].get(col, {}).get("unique_count", 0)
            cat_info.append(f"{col} (unique: {u_count}{sample_str})")
        lines.append(f"- Categorical columns: {'; '.join(cat_info)}")

    if profile["boolean_columns"]:
        lines.append(f"- Boolean columns: {', '.join(profile['boolean_columns'])}")

    if profile["candidate_group_columns"]:
        lines.append(f"- Recommended grouping/filtering columns: {', '.join(profile['candidate_group_columns'])}")

    if profile["potential_identifiers"]:
        id_names = [f"{item['column']} ({item['status']})" for item in profile["potential_identifiers"]]
        lines.append(f"- Potential identifier columns: {', '.join(id_names)}")

    if profile["constant_columns"]:
        lines.append(f"- Constant columns (no variance): {', '.join(profile['constant_columns'])}")

    if profile["all_null_columns"]:
        lines.append(f"- All-null columns: {', '.join(profile['all_null_columns'])}")

    if profile["missing_summary"]:
        missing_cols = [f"{col} ({info['missing_percentage']}%)" for col, info in profile["missing_summary"].items()]
        lines.append(f"- Columns with missing values: {', '.join(missing_cols)}")

    return "\n".join(lines)


def _format_single_dataset_lines(header: str, df: pd.DataFrame) -> list[str] | None:
    profile = build_dataset_profile(df)
    if "error" in profile or profile.get("num_rows", 0) == 0:
        return None

    lines = [
        header,
        f"Shape: {profile['num_rows']} rows, {profile['num_columns']} columns",
        "- Column dtypes: " + ", ".join(
            f"{column}: {dtype}" for column, dtype in profile["column_types"].items()
        ),
    ]

    if profile["numeric_columns"]:
        lines.append(f"- Numeric columns: {', '.join(profile['numeric_columns'])}")

    if profile["datetime_columns"]:
        date_info = []
        for col in profile["datetime_columns"]:
            fmt = profile["date_column_details"].get(col, {}).get("format", "datetime")
            date_info.append(f"{col} (format: {fmt})")
        lines.append(f"- Datetime columns: {', '.join(date_info)}")
        lines.extend(_format_temporal_profile_lines(profile))

    if profile["categorical_columns"]:
        cat_info = []
        for col in profile["categorical_columns"]:
            samples = profile["column_profiles"].get(col, {}).get("sample_values", [])
            sample_str = f", e.g. {samples}" if samples else ""
            u_count = profile["column_profiles"].get(col, {}).get("unique_count", 0)
            cat_info.append(f"{col} (unique: {u_count}{sample_str})")
        lines.append(f"- Categorical columns: {'; '.join(cat_info)}")

    if profile["boolean_columns"]:
        lines.append(f"- Boolean columns: {', '.join(profile['boolean_columns'])}")

    if profile["candidate_group_columns"]:
        lines.append(f"- Recommended grouping/filtering columns: {', '.join(profile['candidate_group_columns'])}")

    if profile["potential_identifiers"]:
        id_names = [f"{item['column']} ({item['status']})" for item in profile["potential_identifiers"]]
        lines.append(f"- Potential identifier columns: {', '.join(id_names)}")

    if profile["constant_columns"]:
        lines.append(f"- Constant columns (no variance): {', '.join(profile['constant_columns'])}")

    if profile["all_null_columns"]:
        lines.append(f"- All-null columns: {', '.join(profile['all_null_columns'])}")

    if profile["missing_summary"]:
        missing_cols = [f"{col} ({info['missing_percentage']}%)" for col, info in profile["missing_summary"].items()]
        lines.append(f"- Columns with missing values: {', '.join(missing_cols)}")

    return lines


def format_datasets_context(
    datasets: dict[str, pd.DataFrame] | None = None,
    derived_datasets: dict[str, pd.DataFrame] | None = None,
) -> str:
    """
    Format contexts for all active and derived datasets, prefixing each with its name.
    """
    if not datasets and not derived_datasets:
        return ""

    contexts = []

    if datasets:
        for name, df in datasets.items():
            lines = _format_single_dataset_lines(f"[Dataset: {name}]", df)
            if lines:
                contexts.append("\n".join(lines))

    if derived_datasets:
        for name, df in derived_datasets.items():
            lines = _format_single_dataset_lines(f"[Derived Dataset: {name}]", df)
            if lines:
                contexts.append("\n".join(lines))

    return "\n\n".join(contexts)


def dataset_info(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return comprehensive structural, semantic, and quality information about the dataset.
    """
    if df is None:
        return {"error": "No dataset is loaded."}
    return build_dataset_profile(df)


DATASET_INFO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "dataset_info",
        "description": (
            "Get comprehensive structural and semantic information about the "
            "currently loaded dataset: number of rows, number of columns, column "
            "names, semantic data types, detected date columns, candidate grouping "
            "columns, missing values, and memory usage. Use this when the user asks "
            "general questions about the dataset's structure or available fields."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "The name of the dataset to analyze (e.g. 'sales.csv'). Optional. Defaults to the primary dataset.",
                }
            },
            "required": [],
        },
    },
}
