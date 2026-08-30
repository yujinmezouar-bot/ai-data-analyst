"""Reproducible behavioral evaluation for the AI Data Analyst."""

from evaluation.benchmark_cases import BENCHMARK_VERSION, BenchmarkCase, benchmark_cases
from evaluation.evaluator import run_benchmark

__all__ = ["BENCHMARK_VERSION", "BenchmarkCase", "benchmark_cases", "run_benchmark"]
