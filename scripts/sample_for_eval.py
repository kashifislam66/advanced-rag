import sys

from app.config.settings import settings
from app.vectorstore.parent_store import ParentStore

count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
parent_store = ParentStore(settings.parent_store_path)

for doc in parent_store.sample(count):
    page = doc.metadata.get("page")
    shown_page = page + 1 if isinstance(page, int) else None
    print("=" * 70)
    print(f"{doc.metadata.get('source')}   page={shown_page}")
    print(doc.page_content[:600])

parent_store.close()