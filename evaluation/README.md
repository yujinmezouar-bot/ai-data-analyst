# V9 Behavioral Evaluation Benchmark

This package evaluates the current AI Data Analyst as an end-to-end system. It measures routing, tool selection, structured numerical results, autonomous execution, ML structure, safety behavior, fallback handling, synthesis checks, and latency across deterministic company-style datasets.

It does **not** prove production readiness, causal correctness, universal prompt understanding, or statistical validity for arbitrary uploaded data. Exact answer wording is deliberately not scored. Numerical and ranking checks use structured evidence; prose checks are limited to important claims or prohibited language.

## Layers

- **Deterministic:** uses controlled provider responses while exercising the real `Agent`, router, tools, autonomous planner boundary, executor, findings, synthesis boundary, and fallback paths. It is reproducible and requires no credentials.
- **Real provider:** uses the configured Groq provider and current production model behavior. It measures actual tool/planner selection, synthesis, provider failures, and latency. It is never run automatically by pytest.

## Commands

```text
python -m evaluation.run_benchmark --layer deterministic
python -m evaluation.run_benchmark --layer real
python -m evaluation.run_benchmark --layer deterministic --case rank_product --no-write
```

Real-provider runs use `GROQ_API_KEY` through the existing provider configuration. If unavailable, the command reports an unavailable run without failing or exposing credentials.

JSON and Markdown review artifacts are written to `evaluation/results/`, which is ignored by Git. Each real-provider case records the question, expectations, routing, tools, bounded evidence, final answer, checks, latency, and errors. Artifacts record the benchmark version, model, provider, timestamp, and case IDs—not secrets.

## Interpreting metrics

Overall pass rate is intentionally separated from routing, tool-selection, numerical, safety, autonomous, and ML rates. Category breakdowns identify where behavior is weak without encouraging prompt-specific production changes. Safe refusal can pass. A failed case should be classified and investigated before changing production behavior.

Known limitations include scripted deterministic tool choice, bounded non-semantic synthesis checks, synthetic datasets, and provider nondeterminism. Human review of real-provider artifacts remains necessary.
