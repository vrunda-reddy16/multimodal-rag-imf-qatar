import numpy as np


def answer_question(vector_store, question, k=3):
    model = vector_store["model"]
    index = vector_store["index"]
    texts = vector_store["texts"]
    pages = vector_store["pages"]

    query_embedding = model.encode([question])
    distances, indices = index.search(np.array(query_embedding), k)

    answer = "Relevant information from the document:\n\n"

    for idx in indices[0]:
        answer += f"(Page {pages[idx]}) {texts[idx][:500]}...\n\n"

    return answer
