from typing import Any

import pandas as pd


# Safety caps, consistent with MAX_GROUPS_RETURNED in tools/groupby.py --
# never dump a full NxN matrix to the LLM.
MAX_PAIRS_RETURNED = 20
DEFAULT_TOP_N = 10


def correlation_analysis(
    df: pd.DataFrame,
    column: str | None = None,
    columns: list[str] | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    """
    Compute correlations between numerical columns.

    - If `column` is given, returns that column's correlation with every
      other numeric column, sorted by strength (most useful for
      "what is correlated with sales?").
    - If `columns` is given, restricts analysis to that subset.
    - Otherwise, returns the strongest pairwise correlations across all
      numeric columns (never the full matrix, to avoid huge payloads).

    Constant columns (zero variance) are dropped since correlation is
    undefined for them. Missing values are handled via pandas' pairwise
    complete-observation correlation.
    """
    if df is None:
        return {"error": "No dataset is loaded."}

    numeric_df = df.select_dtypes(include="number")

    if column is not None and column not in df.columns:
        return {"error": f"Column '{column}' not found in the dataset.", "available_columns": list(df.columns)}

    if column is not None and column not in numeric_df.columns:
        return {"error": f"Column '{column}' is not numeric, so correlation cannot be computed."}

    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return {"error": f"Columns not found in the dataset: {missing}", "available_columns": list(df.columns)}
        non_numeric = [c for c in columns if c not in numeric_df.columns]
        if non_numeric:
            return {"error": f"Columns are not numeric: {non_numeric}"}
        numeric_df = numeric_df[columns]

    if column is not None and column not in numeric_df.columns:
        # Can happen if `columns` was also given but excluded `column`.
        numeric_df = pd.concat([numeric_df, df[[column]]], axis=1)

    # Drop constant columns -- correlation is undefined (NaN) for them
    # and would otherwise just clutter results.
    constant_columns = [c for c in numeric_df.columns if numeric_df[c].nunique(dropna=True) <= 1]
    dropped_note = None
    if constant_columns:
        numeric_df = numeric_df.drop(columns=constant_columns)
        dropped_note = f"Excluded constant column(s) (no variance): {constant_columns}"

    if numeric_df.shape[1] < 2:
        return {
            "error": (
                "Not enough numeric columns with variance to compute correlations "
                f"(found {numeric_df.shape[1]})."
            ),
            "note": dropped_note,
        }

    try:
        corr_matrix = numeric_df.corr(method="pearson")
    except Exception as e:
        return {"error": f"Correlation computation failed: {e}"}

    effective_top_n = top_n or DEFAULT_TOP_N
    effective_top_n = min(effective_top_n, MAX_PAIRS_RETURNED)

    # ------------------------------------------------------------
    # Single target column: correlations vs everything else.
    # ------------------------------------------------------------
    if column is not None:
        if column in constant_columns:
            return {"error": f"Column '{column}' has no variance (constant), so it cannot be correlated with anything."}

        series = corr_matrix[column].drop(labels=[column], errors="ignore")
        series = series.dropna().sort_values(key=lambda s: s.abs(), ascending=False)

        result = {
            str(other): round(float(value), 3)
            for other, value in series.items()
        }

        truncated = len(result) > effective_top_n
        if truncated:
            result = dict(list(result.items())[:effective_top_n])

        pos_items = [(k, v) for k, v in series.items() if v > 0]
        neg_items = [(k, v) for k, v in series.items() if v < 0]

        strongest_pos = max(pos_items, key=lambda kv: kv[1]) if pos_items else None
        strongest_neg = min(neg_items, key=lambda kv: kv[1]) if neg_items else None

        output = {
            "target_column": column,
            "correlations": result,
            "strongest_correlation": next(iter(result.items()), None),
            "strongest_positive": (strongest_pos[0], round(strongest_pos[1], 3)) if strongest_pos else None,
            "strongest_negative": (strongest_neg[0], round(strongest_neg[1], 3)) if strongest_neg else None,
            "analytical_note": "Correlation measures statistical association, not causation.",
        }
        if dropped_note:
            output["note"] = dropped_note
        if truncated:
            output["note"] = (output.get("note", "") + " " if output.get("note") else "") + (
                f"Showing the top {effective_top_n} strongest correlations."
            )
        return output

    # ------------------------------------------------------------
    # No target column: strongest pairwise correlations overall,
    # never the full matrix.
    # ------------------------------------------------------------
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr_matrix.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            value = corr_matrix.loc[col_a, col_b]
            if pd.isna(value):
                continue
            pairs.append((col_a, col_b, float(value)))

    if not pairs:
        return {"error": "No valid correlations could be computed (all pairs had insufficient overlapping data)."}

    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    total_pairs = len(pairs)
    top_pairs = pairs[:effective_top_n]

    pos_pairs = [p for p in pairs if p[2] > 0]
    neg_pairs = [p for p in pairs if p[2] < 0]

    strongest_pos_pair = max(pos_pairs, key=lambda p: p[2]) if pos_pairs else None
    strongest_neg_pair = min(neg_pairs, key=lambda p: p[2]) if neg_pairs else None

    output = {
        "numeric_columns_analyzed": cols,
        "top_correlations": [
            {"column_1": a, "column_2": b, "correlation": round(v, 3)}
            for a, b, v in top_pairs
        ],
        "strongest_positive_pair": {
            "column_1": strongest_pos_pair[0],
            "column_2": strongest_pos_pair[1],
            "correlation": round(strongest_pos_pair[2], 3),
        } if strongest_pos_pair else None,
        "strongest_negative_pair": {
            "column_1": strongest_neg_pair[0],
            "column_2": strongest_neg_pair[1],
            "correlation": round(strongest_neg_pair[2], 3),
        } if strongest_neg_pair else None,
        "analytical_note": "Correlation measures statistical association, not causation.",
    }
    if total_pairs > effective_top_n:
        output["note"] = f"Showing the top {effective_top_n} of {total_pairs} pairwise correlations by strength."
    if dropped_note:
        output["note"] = (output.get("note", "") + " " if output.get("note") else "") + dropped_note

    return output


CORRELATION_ANALYSIS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "correlation_analysis",
        "description": (
            "Compute correlations between numeric columns. Use this for questions like "
            "'what variables are correlated with sales?', 'show the correlation between "
            "price and quantity', or 'give me the strongest correlations in the dataset'. "
            "Set 'column' to see what correlates with one specific numeric column. Set "
            "'columns' to restrict analysis to a specific subset. Leave both unset to get "
            "the overall strongest pairwise correlations across the dataset (never a full "
            "matrix)."
        ),
        "parameters": {
            "type": "object",
            "properties": { "dataset_name": {"type": "string", "description": "The name of the dataset to analyze (e.g. 'sales.csv'). Optional. Defaults to the primary dataset."},
                "column": {
                    "type": ["string", "null"],
                    "description": (
                        "A single numeric column to correlate against all others, e.g. 'sales'. "
                        "Set to null (or omit) to get overall strongest pairwise correlations instead."
                    ),
                },
                "columns": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Restrict the analysis to these numeric columns only. Set to null (or omit) to consider all numeric columns.",
                },
                "top_n": {
                    "type": ["integer", "null"],
                    "description": f"Number of strongest correlations to return. Defaults to {DEFAULT_TOP_N} if null or omitted, capped at {MAX_PAIRS_RETURNED}.",
                },
            },
            "required": [],
        },
    },
}
