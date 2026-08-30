# AI Data Analyst

An evidence-grounded AI data analysis application developed as a final-year university project. It combines large-language-model interpretation and planning with deterministic Python analysis tools, bounded autonomous execution, supervised machine learning, multi-dataset analysis, and reproducible reports.

The LLM decides how to approach a request and explains recorded findings. Numerical calculations are performed by the application's pandas and scikit-learn tools; the LLM is instructed not to invent or recompute values. The system is a bounded analytical assistant, not a fully autonomous data scientist or an enterprise analytics platform.

## Key capabilities

- Upload one or more CSV or Excel (`.xlsx`/`.xls`) datasets.
- Inspect schemas, data types, missing values, date coverage, categorical fields, and identifier hints.
- Calculate descriptive statistics, grouped comparisons, rankings, outliers, and correlations.
- Analyze trends, percentage changes, and day/week/month/quarter/year periods.
- Decompose additive KPI changes into ranked group contributions and offsets.
- Create interactive Plotly visualizations.
- Discover possible relationships between datasets and execute guarded joins.
- Plan and preflight autonomous multi-step analyses against the actual dataset schemas and tool contracts.
- Perform one bounded adaptive review and, when justified, a small validated follow-up investigation.
- Evaluate supervised classification and numeric regression models with leakage and split safeguards.
- Generate deterministic Markdown reports from recorded evidence and provenance.

## Architecture

```text
Streamlit UI
    |
    v
Agent
    |-- Reactive analysis
    |     |-- LLM tool selection
    |     |-- deterministic Python tools
    |     `-- tools-disabled grounded synthesis
    |
    `-- Autonomous analysis
          |-- AnalysisPlanner
          |-- AnalysisPlan validation
          |-- deterministic complete-plan preflight
          |-- Executor
          |-- FindingsStore and provenance
          |-- one bounded adaptive review
          `-- tools-disabled grounded synthesis
```

`Agent.run()` conservatively selects between the existing reactive and autonomous paths. Autonomous plans are grounded with bounded dataset profiles, temporal metadata, relationship summaries, and the same tool schemas used for execution validation. Preflight rejects structurally invalid plans before any plan tool runs. Initial autonomous failures fall back to the reactive path, while completed findings are retained if adaptive review or final synthesis fails.

Deterministic tools remain responsible for calculations, joins, visualizations, ML evaluation, and structured evidence. Reports consume the completed result and do not rerun the analysis.

## Installation

The validated development environment uses Python 3.12 on Windows. A recent compatible Python 3 installation should also work on other platforms.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local `.env` file in the project root. Never commit this file or a real key:

```text
GROQ_API_KEY=your_groq_api_key_here
```

The application currently uses the Groq-backed provider configured in `agent/llm.py`. `.env` is excluded by `.gitignore`.

## Running the application

```text
streamlit run app.py
```

In the Streamlit interface:

1. Upload one or more CSV or Excel files.
2. Review the detected schema, dates, and preview.
3. Ask a question in natural language.
4. View the answer and any generated Plotly visualization.
5. Generate and download a Markdown analysis report from the completed result.

Questions should name the relevant metric, grouping, periods, prediction target, or datasets when those choices are not obvious. Autonomous routing is deliberately conservative and requires no separate UI control.

## Testing

Run the test suite without making external provider calls:

```text
python -m pytest -q
```

The current reviewed baseline is **518 passing tests**. This records the verified project state rather than guaranteeing that the count will never change as tests evolve.

## Behavioral evaluation

The V9 evaluation framework has two layers:

1. **Deterministic controlled-provider evaluation** exercises the real Agent, routing, tools, autonomous boundaries, evidence scoring, ML safeguards, and fallback paths using scripted provider responses.
2. **Real-provider evaluation** exercises the configured Groq model and therefore consumes provider quota and can vary with provider availability and model behavior.

```text
python -m evaluation.run_benchmark --layer deterministic
python -m evaluation.run_benchmark --layer real
```

The real-provider layer is intentionally not part of normal pytest. The current deterministic baseline is **47/47 cases passing**. Generated benchmark artifacts are written below `evaluation/results/` and are ignored by Git.

## Machine learning boundaries

ML V1 supports supervised classification and numeric regression. It compares a deterministic naive baseline with a linear predictive model:

- classification: most-frequent baseline and logistic regression;
- regression: mean baseline and ridge regression.

Numeric and categorical preprocessing, imputation, scaling, one-hot encoding, identifier exclusion, cardinality limits, exact-duplicate leakage rejection, and bounded feature associations are built in. Evaluation supports stratified random splits, latest-row temporal splits, or group-isolated random splits for explicitly named entities. Metrics are held-out estimates, and feature associations are predictive rather than causal.

ML V1 does not target deep learning, forecasting, clustering, hyperparameter search, broad model selection, persisted model deployment, or production prediction serving. Combined grouped-temporal splitting is also outside the current scope.

## Multi-dataset and join safety

Relationship discovery uses column-name similarity, type compatibility, and bounded value overlap to suggest possible links. Join inspection classifies candidate relationships as 1:1, 1:N, N:1, or N:N.

- 1:1, 1:N, and N:1 joins may proceed after validation.
- N:N joins are blocked by default to reduce row-explosion risk.
- Joined results have a maximum row-count guard.
- Successful joins are registered under canonical derived-dataset names for downstream steps.

Derived datasets are currently request-scoped: they can be used by later steps in the same analysis request but are reset when the next `Agent.run()` begins.

## Reports

Report Generation V1 builds a bounded Markdown report from the completed answer, structured findings or reactive evidence, dataset profiles, limitations, methodology, and provenance. Report construction is deterministic: it does not rerun tools and does not make another LLM call. Interactive Plotly charts remain in Streamlit and are described, rather than embedded, in the Markdown download.

## Privacy and data flow

Most analytical computation runs locally in Python, including pandas profiling and analysis, statistics, joins, Plotly figure construction, scikit-learn model evaluation, and Markdown report rendering. Full DataFrames are not intentionally inserted into planner or synthesis prompts.

The application is **not fully offline**. The configured Groq LLM may receive bounded context needed for interpretation, planning, tool selection, adaptive review, or explanation, including:

- the user's question and bounded recent conversation context;
- dataset and column names, schema/type information, and date/period metadata;
- bounded categorical examples and relationship summaries;
- bounded analytical tool results;
- structured findings and provenance required to explain completed analysis.

Because tool evidence can contain category names, outlier examples, filters, or other values, the project does not claim that row-level information can never reach the external provider. Users should not upload confidential or sensitive data unless they accept this external-processing boundary and the configured provider's applicable terms.

This final-year-project implementation does not provide an enterprise authentication, authorization, storage, retention, audit, or security model and is not currently hardened for highly confidential production use. Local reports and ignored evaluation artifacts may also contain analytical evidence and should be handled according to the sensitivity of the source data.

## Known limitations

- Groq availability, quota, and latency affect LLM-guided analysis.
- Autonomous execution is bounded to validated plans and one small adaptive follow-up round; it is not an open-ended agent loop.
- Derived datasets do not persist across separate requests.
- Relationship discovery is heuristic and suggested joins still require runtime safety checks.
- Correlation and contribution decomposition describe observed associations or mathematical contributors, not causal effects.
- ML is limited to the supervised linear-model V1 scope described above and one held-out split.
- The application has no enterprise identity, storage, retention, or multi-user security layer.
- Reports are Markdown rather than formal PDF or DOCX exports, and interactive charts are not embedded.
- Large or very wide uploads remain constrained by local memory, request compaction, and configured analytical limits.

## Project status

The project is final-year-project and demonstration ready with explicit bounded limitations. Its strongest contribution is the separation of LLM interpretation and planning from deterministic, validated analytical execution and structured evidence. It should be treated as a research/demo foundation for further product hardening, not as an enterprise-ready autonomous data-science platform.
