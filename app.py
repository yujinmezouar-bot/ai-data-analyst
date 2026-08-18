import streamlit as st
import pandas as pd

from agent.agent import Agent, MAX_HISTORY_MESSAGES
from tools.date_utils import convert_date_columns


st.set_page_config(page_title="AI Data Analyst", layout="wide")

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

if "detected_dates" not in st.session_state:
    st.session_state.detected_dates = {}


# ============================================================
# DATASET LOADER
# Date detection/conversion is consolidated into a single
# implementation (tools/date_utils.py) -- there is no separate
# date-parsing logic in this file anymore.
# ============================================================

def load_dataset(uploaded_file) -> tuple[pd.DataFrame, dict]:
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")

    df, detected_dates = convert_date_columns(df)
    return df, detected_dates


def get_agent() -> Agent:
    if st.session_state.agent is None:
        st.session_state.agent = Agent()
    return st.session_state.agent


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if (
            "uploaded_filename" not in st.session_state
            or st.session_state.uploaded_filename != uploaded_file.name
        ):
            st.session_state.df, st.session_state.detected_dates = load_dataset(uploaded_file)
            st.session_state.uploaded_filename = uploaded_file.name

            # New dataset = new conversation.
            st.session_state.messages = []
            st.session_state.last_figure = None

        st.success(f"Loaded '{uploaded_file.name}' successfully.")

        if st.session_state.detected_dates:
            date_summary = ", ".join(
                f"{col} ({meta['format']})"
                for col, meta in st.session_state.detected_dates.items()
            )
            st.caption(f"Detected date column(s): {date_summary}")

    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.session_state.df = None


# ============================================================
# DATASET INFORMATION
# ============================================================

df = st.session_state.df

if df is not None:
    st.header("Dataset Information")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Column Names and Data Types")
    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
    })
    st.dataframe(dtypes_df, use_container_width=True)

    st.subheader("Preview (first 5 rows)")
    st.dataframe(df.head(), use_container_width=True)

    # ========================================================
    # CONVERSATION
    # ========================================================

    st.header("Ask a question about your data")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.last_figure = None
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.text_area(
        "Ask your question:",
        placeholder=(
            "Examples:\n"
            "• What is the average Weekly_Sales?\n"
            "• What about by store?\n"
            "• Show me monthly sales for 2024.\n"
            "• Which month had the highest sales?\n"
            "• Show me the top 10 stores by sales.\n"
            "• Compare those stores by month."
        ),
    )

    if st.button("Analyze", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("Thinking..."):
                    agent = get_agent()
                    result = agent.run(
                        question,
                        df,
                        conversation_history=st.session_state.messages,
                    )

                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

                if len(st.session_state.messages) > MAX_HISTORY_MESSAGES:
                    st.session_state.messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]

                st.session_state.last_figure = result["figure"]

                st.rerun()

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    if st.session_state.last_figure is not None:
        st.subheader("Visualization")
        st.plotly_chart(st.session_state.last_figure, use_container_width=True)

else:
    st.info("Upload a CSV or Excel file to get started.")