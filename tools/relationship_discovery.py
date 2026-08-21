from __future__ import annotations

import re
from typing import Any
import pandas as pd


DISCOVER_RELATIONSHIPS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "discover_relationships",
        "description": (
            "Inspect and discover candidate relationships (join keys) across multiple uploaded datasets "
            "using column name similarity, datatype compatibility, and sample value overlap. "
            "Returns ranked recommendations with confidence scores. This tool is read-only and does not perform joins."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "Optional dataset to focus relationship discovery on. If omitted, checks all dataset pairs.",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold between 0.0 and 1.0 (default: 0.4).",
                },
            },
            "required": [],
        },
    },
}


def _normalize_column_name(col: str) -> str:
    """Normalize column name for lexical comparison by stripping punctuation and common ID affixes."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", str(col).lower())
    # Strip common prefixes/suffixes
    cleaned = re.sub(r"^(id_|id|pk_|fk_)", "", cleaned)
    cleaned = re.sub(r"(_id|_pk|_fk|_num|_no|_code|_key|id|num|code|key)$", "", cleaned)
    return cleaned


def _compute_name_similarity(col1: str, col2: str) -> float:
    """Compute lexical similarity score between two column names."""
    c1, c2 = str(col1).lower().strip(), str(col2).lower().strip()
    if c1 == c2:
        return 1.0

    norm1, norm2 = _normalize_column_name(c1), _normalize_column_name(c2)
    if norm1 and norm2 and norm1 == norm2:
        return 0.95

    # Check substring containment
    if (norm1 and norm2) and (norm1 in norm2 or norm2 in norm1):
        shorter, longer = (norm1, norm2) if len(norm1) < len(norm2) else (norm2, norm1)
        if len(shorter) >= 3:
            return 0.75

    # Character bigram Jaccard similarity
    def get_bigrams(s: str) -> set[str]:
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

    bg1, bg2 = get_bigrams(c1), get_bigrams(c2)
    intersection = len(bg1 & bg2)
    union = len(bg1 | bg2)
    return round(intersection / union, 3) if union > 0 else 0.0


def _compute_value_overlap(s1: pd.Series, s2: pd.Series, max_samples: int = 500) -> float:
    """Compute sample value overlap (Jaccard Index) between two series."""
    vals1 = s1.dropna().unique()[:max_samples]
    vals2 = s2.dropna().unique()[:max_samples]

    if len(vals1) == 0 or len(vals2) == 0:
        return 0.0

    # Normalize values to string for cross-type or formatting tolerance
    set1 = {str(v).strip() for v in vals1 if str(v).strip()}
    set2 = {str(v).strip() for v in vals2 if str(v).strip()}

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return round(intersection / union, 3) if union > 0 else 0.0


def _are_types_compatible(s1: pd.Series, s2: pd.Series) -> bool:
    """Check if two series have compatible data types for potential join keys."""
    dtype1 = s1.dtype
    dtype2 = s2.dtype

    if dtype1 == dtype2:
        return True

    # Numeric compatibility (e.g. int64 and float64)
    if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
        return True

    # String / object / categorical compatibility
    is_str1 = pd.api.types.is_string_dtype(s1) or pd.api.types.is_object_dtype(s1)
    is_str2 = pd.api.types.is_string_dtype(s2) or pd.api.types.is_object_dtype(s2)
    if is_str1 and is_str2:
        return True

    return False


def score_key_pair(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> dict[str, Any] | None:
    """Evaluate a candidate join key pair and return scoring metadata."""
    if left_col not in left_df.columns or right_col not in right_df.columns:
        return None

    s_left = left_df[left_col]
    s_right = right_df[right_col]

    # Filter out all-null or constant single-value columns
    if s_left.dropna().nunique() <= 1 or s_right.dropna().nunique() <= 1:
        return None

    if not _are_types_compatible(s_left, s_right):
        return None

    name_sim = _compute_name_similarity(left_col, right_col)
    val_overlap = _compute_value_overlap(s_left, s_right)

    # If name similarity is low and value overlap is zero, not a candidate
    if name_sim < 0.3 and val_overlap == 0.0:
        return None

    # Composite confidence score
    confidence = round(0.4 * name_sim + 0.6 * val_overlap, 2)

    # Cardinality assessment
    left_unique = s_left.is_unique
    right_unique = s_right.is_unique
    if left_unique and right_unique:
        cardinality = "1:1"
    elif left_unique and not right_unique:
        cardinality = "1:N"
    elif not left_unique and right_unique:
        cardinality = "N:1"
    else:
        cardinality = "N:N"

    return {
        "left_column": left_col,
        "right_column": right_col,
        "name_similarity": name_sim,
        "value_overlap": val_overlap,
        "confidence": confidence,
        "cardinality": cardinality,
        "safe_to_join": cardinality != "N:N",
    }


def discover_relationships(
    datasets: dict[str, pd.DataFrame],
    target_dataset: str | None = None,
    min_confidence: float = 0.4,
) -> dict[str, Any]:
    """
    Scan multiple datasets to discover and rank candidate relationships.
    Read-only tool that never executes joins.
    """
    if not datasets or len(datasets) < 2:
        return {
            "status": "info",
            "relationships": [],
            "message": "At least two datasets are required to discover relationships.",
        }

    dataset_names = list(datasets.keys())
    candidate_pairs: list[dict[str, Any]] = []

    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            name_a, name_b = dataset_names[i], dataset_names[j]

            if target_dataset and target_dataset not in (name_a, name_b):
                continue

            df_a, df_b = datasets[name_a], datasets[name_b]

            for col_a in df_a.columns:
                for col_b in df_b.columns:
                    score = score_key_pair(df_a, df_b, col_a, col_b)
                    if score and score["confidence"] >= min_confidence:
                        candidate_pairs.append({
                            "left_dataset": name_a,
                            "left_column": col_a,
                            "right_dataset": name_b,
                            "right_column": col_b,
                            "confidence": score["confidence"],
                            "name_similarity": score["name_similarity"],
                            "value_overlap": score["value_overlap"],
                            "cardinality": score["cardinality"],
                            "safe_to_join": score["safe_to_join"],
                        })

    # Sort candidates by confidence descending
    candidate_pairs.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "status": "success",
        "relationships_found": len(candidate_pairs),
        "relationships": candidate_pairs,
        "hint": (
            "Use execute_join(left_dataset, right_dataset, left_on, right_on) "
            "with any of the safe (1:1, 1:N, N:1) candidate relationships if your analysis requires combining them."
        ),
    }


def build_schema_graph_summary(datasets: dict[str, pd.DataFrame], max_edges: int = 5) -> str:
    """Build a compact, deterministic ASCII/text schema relationship map for prompt context."""
    if not datasets or len(datasets) < 2:
        return ""

    res = discover_relationships(datasets, min_confidence=0.5)
    candidates = res.get("relationships", [])

    if not candidates:
        return ""

    lines = ["[Schema Relationship Map]"]
    for rel in candidates[:max_edges]:
        lines.append(
            f"- {rel['left_dataset']} ({rel['left_column']}) <---[{rel['cardinality']}, {int(rel['confidence']*100)}% match]---> {rel['right_dataset']} ({rel['right_column']})"
        )

    return "\n".join(lines)
