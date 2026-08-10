import streamlit as st
import pandas as pd

from agent.agent import Agent

st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("AI Data Analyst")
st.write("Upload a dataset and ask questions about it in plain English.")

# --- Session state setup ---
# session_state persists values across Streamlit reruns.
# Without this, the DataFrame and the Agent would be recreated every rerun.
if "df" not in st.session_state:
    st.session_state.df = None

if "agent" not in st.session_state:
    st.session_state.agent = None


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
    """Create the Agent once and reuse it across reruns.
    Creating a new Groq client on every rerun is wasteful and unnecessary."""
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

    # --- Question section ---
    st.header("Ask a question about your data")
    question = st.text_area(
        "Ask your question:",
        placeholder="e.g. What is the average salary by department?",
    )

    if st.button("Analyze", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("Thinking..."):
                    agent = get_agent()
                    result = agent.run(question, df)

                st.subheader("AI Answer")
                st.write(result["answer"])

                if result["figure"] is not None:
                    st.subheader("Visualization")
                    st.plotly_chart(result["figure"], use_container_width=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
else:
    st.info("Upload a CSV or Excel file to get started.")