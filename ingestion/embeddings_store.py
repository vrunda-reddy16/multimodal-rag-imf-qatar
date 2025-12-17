import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def create_vector_store(chunks):
    """
    Creates a FAISS vector index using local sentence-transformer embeddings
    """
    texts = [c["text"] for c in chunks]
    pages = [c["page"] for c in chunks]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    vector_store = {
        "index": index,
        "texts": texts,
        "pages": pages,
        "model": model
    }

    return vector_store
