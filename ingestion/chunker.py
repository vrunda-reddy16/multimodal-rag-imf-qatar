from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_pages(pages, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = []

    for page in pages:
        splits = splitter.split_text(page["text"])
        for text in splits:
            chunks.append({
                "page": page["page"],
                "text": text
            })

    return chunks
