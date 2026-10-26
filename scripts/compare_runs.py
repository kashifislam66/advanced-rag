import argparse
import json
import sys

from app.config.settings import settings
from app.evaluation.compare import question_changes, regression_check
from app.evaluation.metrics import average_scores, summarize_records
from app.evaluation.runner import load_records

parser = argparse.ArgumentParser(description="Compare saved evaluation runs side by side.")
parser.add_argument("runs", nargs="+", help="run names; the first is the reference")
parser.add_argument("--tolerance", type=float, default=0.05)
parser.add_argument("--fail-on-regression", action="store_true")
args = parser.parse_args()

loaded = {name: load_records(settings.eval_runs_dir / f"{name}.jsonl") for name in args.runs}
summaries = {name: summarize_records(records) for name, records in loaded.items()}

id_sets = {name: {r.id for r in records} for name, records in loaded.items()}
if len({frozenset(ids) for ids in id_sets.values()}) > 1:
    print("WARNING: runs cover different questions; numbers are not directly comparable.\n")

ROWS = ["questions", "errors", "hit_rate", "mrr", "full_hit_rate", "false_refusal_rate",
        "correct_refusal_rate", "median_latency_ms", "mean_llm_calls"]


def fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


print(f"{'metric':<22}" + "".join(f"{name:>22}" for name in loaded))
for metric in ROWS:
    print(f"{metric:<22}" + "".join(f"{fmt(summaries[name][metric]):>22}" for name in loaded))

answerable = summaries[args.runs[0]]["answerable"]
if answerable:
    print(f"\n1 answerable question = {100 / answerable:.1f} points. Smaller gaps are probably noise.")


def load_ragas(name):
    path = settings.eval_runs_dir / f"{name}.ragas.jsonl"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


ragas = {name: load_ragas(name) for name in loaded}
if any(rows for rows in ragas.values()):
    print("\nRAGAS (mean, failed rows):")
    metric_names = sorted({key for rows in ragas.values() if rows for key in rows[0]} - {"id", "category"})
    print(f"{'metric':<42}" + "".join(f"{name:>22}" for name in loaded))
    for metric in metric_names:
        cells = []
        for name in loaded:
            if not ragas[name]:
                cells.append("-")
                continue
            stats = average_scores(ragas[name], [metric])[metric]
            cells.append(f"{fmt(stats['mean'])} (fail {stats['failed']})")
        print(f"{metric:<42}" + "".join(f"{cell:>22}" for cell in cells))

reference = args.runs[0]
regressed = False
for name in args.runs[1:]:
    problems = regression_check(summaries[reference], summaries[name], args.tolerance)
    changes = question_changes(loaded[reference], loaded[name])
    print(f"\n--- {name} vs {reference} ---")
    print("Regressions beyond tolerance:", problems or "none")
    for question_id, text in changes:
        print(f"  {question_id}: {text}")
    regressed = regressed or bool(problems)

if args.fail_on_regression and regressed:
    sys.exit(1)