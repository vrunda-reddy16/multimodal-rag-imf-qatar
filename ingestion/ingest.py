from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from chunker import chunk_pages
from embeddings_store import create_vector_store
from qa import answer_question


def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


if __name__ == "__main__":
    # Step 1: Extract text
    pages = extract_pdf_text("data/imf_qatar_report.pdf")
    print(f"Extracted {len(pages)} pages")
    print(pages[0]["text"][:500])

    # Step 2: Chunk text
    chunks = chunk_pages(pages)
    print(f"\nCreated {len(chunks)} chunks")
    print(chunks[0]["text"][:300])

    # Step 3: Create vector store
    vector_store = create_vector_store(chunks)
    print("\nVector store created successfully!")

    # Step 4: Ask multiple questions
    questions = [
        "What are the key economic risks facing Qatar according to the IMF report?",
        "What is the IMF’s assessment of Qatar’s economic growth outlook?",
        "What fiscal policy recommendations does the IMF make for Qatar?",
        "How does the IMF evaluate Qatar’s progress on diversification and reforms?"
    ]

    for q in questions:
        answer = answer_question(vector_store, q)

        print("\nQUESTION:")
        print(q)

        print("\nANSWER:")
        print(answer)
