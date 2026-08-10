import streamlit as st
import pandas as pd

from agent.agent import Agent, MAX_HISTORY_MESSAGES

st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("AI Data Analyst")
st.write("Upload a dataset and ask questions about it in plain English.")

# --- Session state setup ---
if "df" not in st.session_state:
    st.session_state.df = None

if "agent" not in st.session_state:
    st.session_state.agent = None

# Conversation memory: plain {"role": "user"/"assistant", "content": str}
# dicts only. Never store tool_calls or role="tool" entries here -- see
# the warning in agent/agent.py for why.
if "messages" not in st.session_state:
    st.session_state.messages = []

# The most recent figure, shown below the chat. Only the latest chart is
# kept (not one per historical turn) to keep session state light.
if "last_figure" not in st.session_state:
    st.session_state.last_figure = None


def load_dataset(uploaded_file) -> pd.DataFrame:
    """Load an uploaded CSV or Excel file into a pandas DataFrame."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")


def get_agent() -> Agent:
    """Create the Agent once and reuse it across reruns."""
    if st.session_state.agent is None:
        st.session_state.agent = Agent()
    return st.session_state.agent


# --- File upload section ---
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is not None:
    try:
        st.session_state.df = load_dataset(uploaded_file)
        st.success(f"Loaded '{uploaded_file.name}' successfully.")
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.session_state.df = None

# --- Dataset info section ---
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

    # --- Conversation section ---
    st.header("Ask a question about your data")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.last_figure = None
            st.rerun()

    # Render prior turns as a chat history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.text_area(
        "Ask your question:",
        placeholder="e.g. What is the average salary by department? Then try a follow-up like 'what about by store?'",
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

                # Persist this turn AFTER the call, so the history sent
                # to the Agent never includes the current question twice.
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

                # Trim history so it can't grow unbounded across a long session.
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
