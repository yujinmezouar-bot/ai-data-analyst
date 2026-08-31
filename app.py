import logging

import streamlit as st
import pandas as pd

from agent.agent import Agent, MAX_HISTORY_MESSAGES
from reports.report_builder import build_analysis_report, render_markdown
from ui_utils import (
    build_analysis_details,
    dataset_signature,
    load_dataset,
    safe_error_diagnostic,
    user_error_message,
)


LOGGER = logging.getLogger("ai_data_analyst.app")


st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("AI Data Analyst")
st.write(
    "Ask questions in plain English using LLM-guided planning and deterministic "
    "analytical tools."
)
st.info(
    "Privacy: analytical calculations primarily run locally. Bounded dataset metadata, "
    "context, and analytical results may be sent to the configured Groq LLM for "
    "interpretation, planning, and explanation. Do not upload confidential or sensitive "
    "data unless you accept this external-provider processing boundary."
)


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "datasets" not in st.session_state:
    st.session_state.datasets = {}

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_figure" not in st.session_state:
    st.session_state.last_figure = None

if "detected_dates" not in st.session_state:
    st.session_state.detected_dates = {}

if "last_analysis_result" not in st.session_state:
    st.session_state.last_analysis_result = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "last_report_markdown" not in st.session_state:
    st.session_state.last_report_markdown = None


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

uploaded_files = st.file_uploader("Upload your dataset(s)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

if uploaded_files:
    try:
        # Build signature from all files to detect changes
        upload_signature = tuple(dataset_signature(f) for f in uploaded_files)
        if (
            "uploaded_signature" not in st.session_state
            or st.session_state.uploaded_signature != upload_signature
        ):
            st.session_state.datasets = {}
            st.session_state.detected_dates = {}
            for f in uploaded_files:
                df, dates = load_dataset(f)
                st.session_state.datasets[f.name] = df
                st.session_state.detected_dates.update(dates)

            # Set primary df for backward compatible UI preview
            st.session_state.df = next(iter(st.session_state.datasets.values())) if st.session_state.datasets else None
            st.session_state.uploaded_signature = upload_signature

            # New dataset = new conversation.
            st.session_state.messages = []
            st.session_state.last_figure = None
            st.session_state.last_analysis_result = None
            st.session_state.last_question = None
            st.session_state.last_report_markdown = None

        names = ", ".join(st.session_state.datasets.keys())
        st.success(f"Loaded successfully: {names}")

        if st.session_state.detected_dates:
            date_summary = ", ".join(
                f"{col} ({meta['format']})"
                for col, meta in st.session_state.detected_dates.items()
            )
            st.caption(f"Detected date column(s): {date_summary}")

    except Exception as e:
        st.error(user_error_message(e, action="upload") if not isinstance(e, ValueError) else str(e))
        st.session_state.df = None
        st.session_state.datasets = {}
        st.session_state.last_figure = None
        st.session_state.last_analysis_result = None
        st.session_state.last_question = None
        st.session_state.last_report_markdown = None


# ============================================================
# DATASET INFORMATION
# ============================================================

df = st.session_state.df

if len(st.session_state.datasets) > 1:
    preview_dataset_name = st.selectbox(
        "Dataset to inspect",
        options=list(st.session_state.datasets.keys()),
        help="Choose which uploaded dataset is shown in the profile and preview below.",
    )
    df = st.session_state.datasets[preview_dataset_name]

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
            st.session_state.last_analysis_result = None
            st.session_state.last_question = None
            st.session_state.last_report_markdown = None
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
                st.session_state.last_analysis_result = None
                st.session_state.last_question = None
                st.session_state.last_report_markdown = None
                with st.spinner("Analyzing your dataset, running tools if needed, and generating the final answer..."):
                    agent = get_agent()
                    result = agent.run(
                        question,
                        df=None,
                        conversation_history=st.session_state.messages,
                        datasets=st.session_state.datasets,
                        autonomous=None,
                    )

                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

                if len(st.session_state.messages) > MAX_HISTORY_MESSAGES:
                    st.session_state.messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]

                st.session_state.last_figure = result["figure"]
                st.session_state.last_analysis_result = result
                st.session_state.last_question = question

                st.rerun()

            except Exception as e:
                st.session_state.last_figure = None
                st.session_state.last_analysis_result = None
                st.session_state.last_question = None
                st.session_state.last_report_markdown = None
                LOGGER.error("Analysis failed: %s", safe_error_diagnostic(e))
                st.error(user_error_message(e))

    if st.session_state.last_figure is not None:
        st.subheader("Visualization")
        st.plotly_chart(st.session_state.last_figure, use_container_width=True)

    if st.session_state.last_analysis_result is not None:
        details = build_analysis_details(st.session_state.last_analysis_result)
        with st.expander("Analysis details", expanded=False):
            st.markdown(f"**Mode:** {details['mode']}")
            tools_text = ", ".join(details["tools"]) if details["tools"] else "No analytical tools recorded"
            st.markdown(f"**Tools used:** {tools_text}")
            st.markdown(f"**Recorded findings/evidence:** {details['finding_count']}")
            if details["initial_plan_steps"] is not None:
                st.markdown(f"**Initial autonomous plan:** {details['initial_plan_steps']} step(s)")
            if details["adaptive_follow_up"]:
                st.markdown(
                    f"**Adaptive follow-up:** completed "
                    f"{details['adaptive_steps_executed']} validated step(s)"
                )
            if details["limitations"]:
                st.markdown("**Safety and limitations**")
                for limitation in details["limitations"]:
                    st.caption(f"• {limitation}")

        st.subheader("Analysis Report")
        if st.button("Generate Report"):
            report = build_analysis_report(
                st.session_state.last_question or "Analysis",
                st.session_state.last_analysis_result,
                st.session_state.datasets,
            )
            st.session_state.last_report_markdown = render_markdown(report)

        if st.session_state.last_report_markdown:
            st.markdown(st.session_state.last_report_markdown)
            st.download_button(
                "Download Markdown Report",
                data=st.session_state.last_report_markdown,
                file_name="ai_data_analysis_report.md",
                mime="text/markdown",
            )

else:
    st.info("Upload a CSV or Excel file to get started.")
