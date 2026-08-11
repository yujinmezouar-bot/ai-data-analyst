import streamlit as st
import pandas as pd
import re

from agent.agent import Agent, MAX_HISTORY_MESSAGES


st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide",
)

st.title("AI Data Analyst")
st.write("Upload a dataset and ask questions about it in plain English.")


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_figure" not in st.session_state:
    st.session_state.last_figure = None


# ============================================================
# DATE DETECTION
# ============================================================

def detect_and_convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically detect columns that contain dates and convert
    them to pandas datetime.

    Supported examples:

        2025-08-10
        2025/08/10
        08-10-2025
        10-08-2025
        08/10/2025
        10/08/2025
        2025-08-10 14:30:00

    The function tries to determine whether a numeric date is
    month-first or day-first by looking for unambiguous values.

    If all values are ambiguous (e.g. 03-04-2025), month-first
    is used as the default.
    """

    df = df.copy()

    # Words frequently found in date column names
    date_name_keywords = [
        "date",
        "day",
        "month",
        "year",
        "time",
        "timestamp",
        "datetime",
    ]

    def looks_like_date_column(column_name: str, series: pd.Series) -> bool:
        """
        Decide whether a column is likely to contain dates.
        """

        column_name_lower = str(column_name).lower()

        # Strong signal from column name
        if any(keyword in column_name_lower for keyword in date_name_keywords):
            return True

        # Only inspect object/string columns here
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return False

        sample = series.dropna().astype(str).str.strip()

        if sample.empty:
            return False

        sample = sample.head(100)

        # Common date patterns
        date_pattern = re.compile(
            r"""
            ^
            (
                \d{4}[-/]\d{1,2}[-/]\d{1,2}
                |
                \d{1,2}[-/]\d{1,2}[-/]\d{4}
                |
                \d{4}\.\d{1,2}\.\d{1,2}
                |
                \d{1,2}\.\d{1,2}\.\d{4}
            )
            (?:\s+\d{1,2}:\d{2}(?::\d{2})?)?
            $
            """,
            re.VERBOSE,
        )

        matches = sample.apply(
            lambda x: bool(date_pattern.match(x))
        )

        # At least 70% of sampled values should look like dates
        return matches.mean() >= 0.70

    def detect_date_format(series: pd.Series) -> str:
        """
        Detect day-first vs month-first for dates such as:

            03-15-2025 -> clearly month-first
            15-03-2025 -> clearly day-first

        If all dates are ambiguous, month-first is used.
        """

        values = (
            series.dropna()
            .astype(str)
            .str.strip()
            .head(1000)
        )

        day_first_evidence = 0
        month_first_evidence = 0

        for value in values:

            # Extract date part
            match = re.match(
                r"^(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})",
                value,
            )

            if not match:
                continue

            first = int(match.group(1))
            second = int(match.group(2))
            third = int(match.group(3))

            # ISO format: YYYY-MM-DD
            if first >= 1000:
                continue

            # We only care about formats like:
            # DD-MM-YYYY
            # MM-DD-YYYY

            if third < 1000:
                continue

            # Example:
            # 15-03-2025 -> day first
            if first > 12 and second <= 12:
                day_first_evidence += 1

            # Example:
            # 03-15-2025 -> month first
            elif second > 12 and first <= 12:
                month_first_evidence += 1

        if day_first_evidence > month_first_evidence:
            return "dayfirst"

        if month_first_evidence > day_first_evidence:
            return "monthfirst"

        # Ambiguous case:
        # default to MM-DD-YYYY as requested
        return "monthfirst"

    for column in df.columns:

        series = df[column]

        if not looks_like_date_column(column, series):
            continue

        try:

            format_preference = detect_date_format(series)

            if format_preference == "dayfirst":
                converted = pd.to_datetime(
                    series,
                    errors="coerce",
                    dayfirst=True,
                )
            else:
                converted = pd.to_datetime(
                    series,
                    errors="coerce",
                    dayfirst=False,
                )

            original_non_null = series.notna().sum()

            converted_non_null = converted.notna().sum()

            if original_non_null == 0:
                continue

            conversion_rate = (
                converted_non_null / original_non_null
            )

            # Only replace the column if conversion was successful
            if conversion_rate >= 0.70:

                df[column] = converted

        except Exception:
            # Never prevent the dataset from loading because
            # date detection failed.
            continue

    return df


# ============================================================
# DATASET LOADER
# ============================================================

def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load CSV or Excel file and automatically detect date columns.
    """

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    elif filename.endswith((".xlsx", ".xls")):

        df = pd.read_excel(uploaded_file)

    else:

        raise ValueError(
            "Unsupported file type. "
            "Please upload a .csv or .xlsx file."
        )

    # Automatically detect and convert dates
    df = detect_and_convert_dates(df)

    return df


# ============================================================
# AGENT
# ============================================================

def get_agent() -> Agent:

    if st.session_state.agent is None:
        st.session_state.agent = Agent()

    return st.session_state.agent


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
)


if uploaded_file is not None:

    try:

        # Avoid unnecessary reloads
        # when Streamlit reruns the application.
        if (
            "uploaded_filename" not in st.session_state
            or st.session_state.uploaded_filename
            != uploaded_file.name
        ):

            st.session_state.df = load_dataset(
                uploaded_file
            )

            st.session_state.uploaded_filename = (
                uploaded_file.name
            )

            # New dataset = new conversation
            st.session_state.messages = []
            st.session_state.last_figure = None

        st.success(
            f"Loaded '{uploaded_file.name}' successfully."
        )

    except Exception as e:

        st.error(
            f"Failed to load file: {e}"
        )

        st.session_state.df = None


# ============================================================
# DATASET INFORMATION
# ============================================================

df = st.session_state.df


if df is not None:

    st.header("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Rows",
            df.shape[0],
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1],
        )

    st.subheader(
        "Column Names and Data Types"
    )

    dtypes_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
        }
    )

    st.dataframe(
        dtypes_df,
        use_container_width=True,
    )

    st.subheader(
        "Preview (first 5 rows)"
    )

    st.dataframe(
        df.head(),
        use_container_width=True,
    )


    # ========================================================
    # CONVERSATION
    # ========================================================

    st.header(
        "Ask a question about your data"
    )

    col_clear, _ = st.columns([1, 5])

    with col_clear:

        if st.button(
            "Clear conversation"
        ):

            st.session_state.messages = []

            st.session_state.last_figure = None

            st.rerun()


    # Display previous messages

    for msg in st.session_state.messages:

        with st.chat_message(
            msg["role"]
        ):

            st.write(
                msg["content"]
            )


    question = st.text_area(
        "Ask your question:",
        placeholder=(
            "Examples:\n"
            "• What are the sales over time?\n"
            "• Show me a line chart of sales by date.\n"
            "• What was the average sales in each month?\n"
            "• Show me sales by store."
        ),
    )


    # ========================================================
    # ANALYZE
    # ========================================================

    if st.button(
        "Analyze",
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            try:

                with st.spinner(
                    "Thinking..."
                ):

                    agent = get_agent()

                    result = agent.run(
                        question,
                        df,
                        conversation_history=(
                            st.session_state.messages
                        ),
                    )


                # Save conversation

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": question,
                    }
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                    }
                )


                # Limit history

                if (
                    len(st.session_state.messages)
                    > MAX_HISTORY_MESSAGES
                ):

                    st.session_state.messages = (
                        st.session_state.messages[
                            -MAX_HISTORY_MESSAGES:
                        ]
                    )


                # Save latest chart

                st.session_state.last_figure = (
                    result["figure"]
                )


                st.rerun()


            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


    # ========================================================
    # DISPLAY CHART
    # ========================================================

    if (
        st.session_state.last_figure
        is not None
    ):

        st.subheader(
            "Visualization"
        )

        st.plotly_chart(
            st.session_state.last_figure,
            use_container_width=True,
        )


else:

    st.info(
        "Upload a CSV or Excel file to get started."
    )
