from __future__ import annotations

import argparse
import json

from evaluation.evaluator import run_benchmark, write_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V9 behavioral evaluation benchmark.")
    parser.add_argument("--layer", choices=["deterministic", "real"], default="deterministic")
    parser.add_argument("--case", action="append", dest="case_ids", help="Run only this case ID; repeatable.")
    parser.add_argument("--no-write", action="store_true", help="Do not write JSON/Markdown artifacts.")
    args = parser.parse_args()

    run = run_benchmark(args.layer, set(args.case_ids) if args.case_ids else None)
    print(json.dumps(run["summary"], indent=2))
    if run["status"] == "unavailable":
        print(run["reason"])
        return 0
    if not args.no_write:
        json_path, markdown_path = write_artifacts(run)
        print(f"JSON: {json_path}")
        print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
