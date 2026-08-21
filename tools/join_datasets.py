import pandas as pd
from typing import Any


INSPECT_JOIN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "inspect_join_viability",
        "description": "Evaluate a potential join between two datasets to determine if it is safe and viable. Use this before executing a join.",
        "parameters": {
            "type": "object",
            "properties": {
                "left_dataset": {
                    "type": "string",
                    "description": "The name of the left dataset.",
                },
                "right_dataset": {
                    "type": "string",
                    "description": "The name of the right dataset.",
                },
                "left_on": {
                    "type": "string",
                    "description": "The column name in the left dataset to join on.",
                },
                "right_on": {
                    "type": "string",
                    "description": "The column name in the right dataset to join on.",
                },
            },
            "required": ["left_dataset", "right_dataset", "left_on", "right_on"],
        },
    },
}

EXECUTE_JOIN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_join",
        "description": "Execute a join between two datasets. Returns metadata and registers the joined dataset. Only use this if you have verified it is safe or if it is a simple 1:1 or 1:N join.",
        "parameters": {
            "type": "object",
            "properties": {
                "left_dataset": {
                    "type": "string",
                    "description": "The name of the left dataset.",
                },
                "right_dataset": {
                    "type": "string",
                    "description": "The name of the right dataset.",
                },
                "left_on": {
                    "type": "string",
                    "description": "The column name in the left dataset to join on.",
                },
                "right_on": {
                    "type": "string",
                    "description": "The column name in the right dataset to join on.",
                },
                "how": {
                    "type": "string",
                    "description": "The type of join to perform: 'inner', 'left', 'right', or 'outer'. Defaults to 'inner'.",
                    "enum": ["inner", "left", "right", "outer"]
                },
            },
            "required": ["left_dataset", "right_dataset", "left_on", "right_on"],
        },
    },
}

MAX_ROW_COUNT_LIMIT = 5000000


def _evaluate_cardinality(left_df: pd.DataFrame, right_df: pd.DataFrame, left_on: str, right_on: str) -> dict[str, Any]:
    if left_on not in left_df.columns:
        return {"error": f"Column '{left_on}' not found in left dataset.", "safe_to_join": False}
    if right_on not in right_df.columns:
        return {"error": f"Column '{right_on}' not found in right dataset.", "safe_to_join": False}
        
    left_dtype = left_df[left_on].dtype
    right_dtype = right_df[right_on].dtype
    if left_dtype != right_dtype:
        return {
            "error": f"Incompatible data types: '{left_on}' is {left_dtype}, but '{right_on}' is {right_dtype}.",
            "safe_to_join": False,
            "hint": "Consider casting or deriving a new column."
        }

    left_unique = left_df[left_on].is_unique
    right_unique = right_df[right_on].is_unique
    
    if left_unique and right_unique:
        cardinality = "1:1"
        safe = True
    elif left_unique and not right_unique:
        cardinality = "1:N"
        safe = True
    elif not left_unique and right_unique:
        cardinality = "N:1"
        safe = True
    else:
        cardinality = "N:N"
        safe = False
        
    return {
        "cardinality": cardinality,
        "left_rows": len(left_df),
        "right_rows": len(right_df),
        "left_missing_keys": int(left_df[left_on].isna().sum()),
        "right_missing_keys": int(right_df[right_on].isna().sum()),
        "safe_to_join": safe,
        "hint": "N:N joins are blocked by default to prevent row explosions." if not safe else "Join is safe to proceed."
    }

def inspect_join_viability(
    left_df: pd.DataFrame, right_df: pd.DataFrame, left_on: str, right_on: str
) -> dict[str, Any]:
    """Tool to inspect if a join is viable."""
    return _evaluate_cardinality(left_df, right_df, left_on, right_on)

def execute_join(
    left_df: pd.DataFrame, right_df: pd.DataFrame, left_on: str, right_on: str, how: str = "inner"
) -> dict[str, Any]:
    """
    Execute the actual join. Returns a dictionary with either "error" or 
    a "dataframe" key containing the actual joined pd.DataFrame plus metadata.
    """
    if how not in ["inner", "left", "right", "outer"]:
        return {"error": f"Invalid join type '{how}'. Must be one of: inner, left, right, outer."}

    eval_meta = _evaluate_cardinality(left_df, right_df, left_on, right_on)
    if not eval_meta.get("safe_to_join", False):
        cardinality = eval_meta.get("cardinality", "unknown")
        default_msg = f"Cardinality is {cardinality}. N:N joins are blocked."
        return {"error": f"Join rejected. {eval_meta.get('error', default_msg)}"}

    try:
        joined_df = pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=how)
    except Exception as e:
        return {"error": f"Join failed during execution: {e}"}

    if len(joined_df) > MAX_ROW_COUNT_LIMIT:
        return {"error": f"Join rejected: Resulting dataframe exceeds maximum row limit of {MAX_ROW_COUNT_LIMIT}."}

    return {
        "status": "success",
        "dataframe": joined_df,
        "shape": list(joined_df.shape),
        "columns": list(joined_df.columns),
        "cardinality": eval_meta.get("cardinality")
    }
