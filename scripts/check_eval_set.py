import sys

from app.config.settings import settings
from app.evaluation.dataset import (
    coverage_warnings,
    find_unknown_sources,
    load_eval_set,
    summarize,
)
from app.utils.logger import setup_logging
from app.vectorstore.parent_store import ParentStore

setup_logging(settings.log_level)

try:
    items = load_eval_set(settings.eval_set_path)
except (FileNotFoundError, ValueError) as exc:
    print(exc)
    sys.exit(1)

parent_store = ParentStore(settings.parent_store_path)
known_sources = parent_store.list_sources()
parent_store.close()

print("Summary:", summarize(items))

for warning in coverage_warnings(items):
    print("WARNING:", warning)

unknown = find_unknown_sources(items, known_sources)
if unknown:
    print("\nERROR: these reference sources are not in the ingested corpus:")
    for question_id, missing in unknown.items():
        print(f"  {question_id}: {missing}")
    print(f"\nIngested sources: {sorted(known_sources)}")
    sys.exit(1)

print("\nEvaluation set looks consistent.")