from app.config.settings import settings
from app.ingestion.loaders import load_documents
from app.utils.logger import setup_logging

setup_logging(settings.log_level)

docs = load_documents(settings.raw_data_dir)
for d in docs[:5]:
    print(d.metadata)
    print(repr(d.page_content[:150]))
    print("-" * 40)