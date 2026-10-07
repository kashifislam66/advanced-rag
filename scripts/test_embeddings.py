import numpy as np

from app.config.settings import settings
from app.embeddings.embedder import (
    embed_query,
    embed_texts,
    get_embedding_dimension,
    get_embedding_model,
)
from app.utils.logger import setup_logging

setup_logging(settings.log_level)

model = get_embedding_model(
    settings.embedding_model_name,
    settings.embedding_device,
    settings.embedding_batch_size,
)
print("Dimension:", get_embedding_dimension(model))

docs = [
    "Employees receive 20 days of paid annual leave.",
    "The office Wi-Fi password changes every month.",
    "Staff can take twenty days of vacation each year.",
]
vectors = np.array(embed_texts(model, docs, settings.embedding_batch_size))
query = np.array(embed_query(model, "How many holidays do employees get?"))

scores = vectors @ query
for text, score in sorted(zip(docs, scores), key=lambda x: -x[1]):
    print(f"{score:.3f}  {text}")