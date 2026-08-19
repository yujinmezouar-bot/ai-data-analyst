import streamlit as st
import pandas as pd

from agent.agent import Agent, MAX_HISTORY_MESSAGES
from ui_utils import dataset_signature, load_dataset, user_error_message


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
# ============================================================

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
        upload_signature = dataset_signature(uploaded_file)
        if (
            "uploaded_signature" not in st.session_state
            or st.session_state.uploaded_signature != upload_signature
        ):
            st.session_state.df, st.session_state.detected_dates = load_dataset(uploaded_file)
            st.session_state.uploaded_signature = upload_signature

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
        st.error(user_error_message(e, action="upload") if not isinstance(e, ValueError) else str(e))
        st.session_state.df = None
        st.session_state.last_figure = None


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

    if df.select_dtypes(include="number").empty:
        st.warning("This dataset has no numeric columns. Structural and missing-data questions are still available.")

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
                st.session_state.last_figure = None
                with st.spinner("Analyzing your dataset, running tools if needed, and generating the final answer..."):
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
                st.session_state.last_figure = None
                st.error(user_error_message(e))

    if st.session_state.last_figure is not None:
        st.subheader("Visualization")
        st.plotly_chart(st.session_state.last_figure, use_container_width=True)

else:
    st.info("Upload a CSV or Excel file to get started.")
