from typing import Any

import pandas as pd

from tools.date_utils import detect_date_columns


MAX_CATEGORY_SAMPLES = 5
IDENTIFIER_KEYWORDS = (
    "id", "identifier", "code", "key", "uuid", "guid", "ssn", "account", "pk", "fk",
    "customer_id", "order_id", "product_id", "user_id", "store_id", "employee_id",
)


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
        "column_profiles": column_profiles,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }


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
            "properties": {},
            "required": [],
        },
    },
}