import argparse
import json
import sys

from app.bootstrap import build_container
from app.config.settings import settings
from app.evaluation.dataset import find_unknown_sources, load_eval_set
from app.evaluation.metrics import average_scores, summarize_by_category, summarize_records
from app.evaluation.ragas_eval import eligible_for_ragas, run_ragas
from app.evaluation.runner import run_system, save_records
from app.evaluation.variants import VARIANT_NAMES, build_variants
from app.utils.logger import setup_logging

parser = argparse.ArgumentParser(description="Run one RAG variant over the evaluation set.")
parser.add_argument("--variant", choices=VARIANT_NAMES, default="hybrid_rerank")
parser.add_argument("--name", default="", help="run name (default: the variant name)")
parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
parser.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all)")
parser.add_argument("--ragas", action="store_true", help="also run the costly RAGAS metrics")
args = parser.parse_args()

setup_logging(settings.log_level)
items = load_eval_set(settings.eval_set_path)
if args.split != "all":
    items = [item for item in items if item.split == args.split]
if args.limit > 0:
    items = items[: args.limit]
if not items:
    sys.exit("No questions selected.")
if args.split in ("test", "all"):
    print("NOTE: the test split is for the final report only. Do not tune on it.\n")

app = build_container(settings)
unknown = find_unknown_sources(items, app.resources.parent_store.list_sources())
if unknown:
    app.close()
    sys.exit(f"Reference sources missing from the corpus: {unknown}")

system = build_variants(app, settings)[args.variant]
run_name = args.name or args.variant


def fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


records = run_system(items, system)
save_records(records, settings.eval_runs_dir / f"{run_name}.jsonl")

columns = ["answerable", "hit_rate", "mrr", "full_hit_rate", "false_refusal_rate",
           "unanswerable", "correct_refusal_rate", "median_latency_ms",
           "mean_llm_calls", "errors"]
print(f"\n=== {run_name} ({len(records)} questions) ===")
print({key: fmt(summarize_records(records)[key]) for key in columns})
for category, summary in summarize_by_category(records).items():
    print(f"{category:<13}", {key: fmt(summary[key]) for key in columns[:5]})

if args.ragas:
    eligible = eligible_for_ragas(records)
    print(f"\n=== RAGAS on {len(eligible)} of {len(records)} questions ===")
    rows, names = run_ragas(
        eligible, app.eval_judge_llm, app.embedder,
        settings.eval_ragas_workers, settings.eval_ragas_timeout,
    )
    ragas_path = settings.eval_runs_dir / f"{run_name}.ragas.jsonl"
    with ragas_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name, stats in average_scores(rows, names).items():
        print(f"{name:<40} mean={fmt(stats['mean'])}  scored={stats['scored']}  failed={stats['failed']}")

app.close()