from io import BytesIO

import pandas as pd
import pytest

from ui_utils import dataset_signature, load_dataset, safe_error_diagnostic, user_error_message


class Upload(BytesIO):
    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name
        self.size = len(content)


def test_load_dataset_supports_realistic_csv_and_date_detection():
    upload = Upload(b"Date,Store,Sales\n2024-01-01,20,100\n2024-02-01,4,90\n", "sales.csv")

    df, dates = load_dataset(upload)

    assert list(df.columns) == ["Date", "Store", "Sales"]
    assert "Date" in dates
    assert df["Store"].tolist() == [20, 4]


def test_load_dataset_supports_excel():
    content = BytesIO()
    pd.DataFrame({"Date": ["2024-01-01"], "Sales": [100]}).to_excel(content, index=False)

    df, dates = load_dataset(Upload(content.getvalue(), "sales.xlsx"))

    assert df.shape == (1, 2)
    assert "Date" in dates


@pytest.mark.parametrize(
    ("upload", "message"),
    [
        (Upload(b"", "empty.csv"), "empty"),
        (Upload(b"", "empty.xlsx"), "empty"),
        (Upload(b"not,a,valid\xffcsv", "broken.csv"), "could not be read"),
        (Upload(b"not an excel workbook", "broken.xlsx"), "could not be read"),
        (Upload(b"", "dataset.txt"), "Unsupported file type"),
    ],
)
def test_load_dataset_returns_clear_upload_errors(upload: Upload, message: str):
    with pytest.raises(ValueError, match=message):
        load_dataset(upload)


def test_load_dataset_rejects_header_only_file():
    with pytest.raises(ValueError, match="no data rows"):
        load_dataset(Upload(b"Store,Sales\n", "header_only.csv"))


def test_dataset_signature_prevents_same_filename_stale_data():
    old_upload = Upload(b"Store,Sales\n20,100\n", "sales.csv")
    replacement = Upload(b"Store,Sales\n20,200\n", "sales.csv")

    assert dataset_signature(old_upload) != dataset_signature(replacement)


def test_user_error_messages_are_safe_and_actionable():
    assert "too large" in user_error_message(RuntimeError("413 Request too large")).lower()
    assert "temporarily unavailable" in user_error_message(RuntimeError("Groq API timeout")).lower()
    assert "could not be completed" in user_error_message(RuntimeError("internal details"))


def test_safe_error_diagnostic_preserves_type_and_cause_but_redacts_secrets():
    try:
        try:
            raise TimeoutError("connection timed out token=private-token")
        except TimeoutError as cause:
            raise RuntimeError("Groq API call failed api_key=private-key") from cause
    except RuntimeError as error:
        diagnostic = safe_error_diagnostic(error)

    assert "RuntimeError" in diagnostic
    assert "TimeoutError" in diagnostic
    assert "private-key" not in diagnostic
    assert "private-token" not in diagnostic
    assert "[redacted]" in diagnostic
