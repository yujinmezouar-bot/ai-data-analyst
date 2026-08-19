"""Small, testable helpers for Streamlit upload and user-facing errors."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from tools.date_utils import convert_date_columns


def load_dataset(uploaded_file: Any) -> tuple[pd.DataFrame, dict]:
    """Load supported uploads and turn parser failures into clear user messages."""
    filename = getattr(uploaded_file, "name", "").lower()

    if filename.endswith(".csv"):
        reader = pd.read_csv
    elif filename.endswith((".xlsx", ".xls")):
        reader = pd.read_excel
    else:
        raise ValueError(
            "Unsupported file type. Please upload a CSV or Excel (.xlsx/.xls) file."
        )

    if getattr(uploaded_file, "size", None) == 0:
        raise ValueError(
            "The uploaded file is empty. Please upload a file with a header and data rows."
        )

    try:
        df = reader(uploaded_file)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "The uploaded file is empty. Please upload a file with a header and data rows."
        ) from exc
    except (pd.errors.ParserError, ValueError, OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            "The uploaded file could not be read. Check that it is a valid CSV or Excel file."
        ) from exc

    if df.shape[1] == 0:
        raise ValueError("The uploaded dataset has no usable columns.")
    if df.empty:
        raise ValueError("The uploaded dataset has no data rows.")

    return convert_date_columns(df)

def dataset_signature(uploaded_file: Any) -> tuple[str, int | None, str]:
    """Identify replacement uploads even when a user reuses the same filename."""
    content = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else b""
    return (
        str(getattr(uploaded_file, "name", "")),
        getattr(uploaded_file, "size", None),
        hashlib.sha256(content).hexdigest(),
    )


def user_error_message(error: Exception, action: str = "analysis") -> str:
    """Avoid exposing provider internals while keeping recovery guidance actionable."""
    detail = str(error).lower()
    if "413" in detail or "request too large" in detail or "tpm limit" in detail:
        return "This request is too large to process. Try a shorter question or clear older conversation messages."
    if any(term in detail for term in ("api", "rate limit", "timeout", "connection", "groq")):
        return "The analysis service is temporarily unavailable. Please try again in a moment."
    if action == "upload":
        return "The file could not be loaded. Check its format and contents, then try again."
    return "The analysis could not be completed. Please try a more specific question."
