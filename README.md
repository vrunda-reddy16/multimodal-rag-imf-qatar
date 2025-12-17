# Multimodal RAG over IMF Qatar Article IV Report

This project implements a **Retrieval-Augmented Generation (RAG)** system for answering natural language questions over the *IMF Qatar 2024 Article IV Consultation Report*.  
The system ingests a long-form economic policy document, performs semantic retrieval using vector embeddings, and returns grounded answers with page references.

---

## Features

- PDF ingestion with page-level metadata
- Overlapping text chunking to preserve context
- Semantic search using sentence-transformer embeddings
- Fast similarity search using FAISS
- Grounded question answering with source references
- Interactive Streamlit-based user interface
- Fully local and reproducible (no external API dependency)

---

## System Architecture

The system follows a standard Retrieval-Augmented Generation (RAG) pipeline:

1. PDF ingestion and text extraction  
2. Text chunking with overlap  
3. Embedding generation using sentence-transformers  
4. Vector indexing using FAISS  
5. Query-time semantic retrieval  
6. Grounded answer construction  
7. Interactive querying via Streamlit UI  

---

##  Project Structure
multimodal-rag/
│
├── data/
│ └── imf_qatar_report.pdf
│
├── ingestion/
│ ├── ingest.py
│ ├── chunker.py
│ ├── embeddings_store.py
│ └── qa.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md


---

## Module Description

- **ingest.py** – Orchestrates the end-to-end ingestion, chunking, and indexing pipeline  
- **chunker.py** – Splits extracted text into overlapping chunks  
- **embeddings_store.py** – Generates embeddings and builds the FAISS vector index  
- **qa.py** – Handles query embedding, retrieval, and answer construction  
- **streamlit_app.py** – Provides an interactive UI for user queries  

---

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt

## Running the Application
#Step 1: Run ingestion and indexing
python ingestion/ingest.py

#Step 2: Launch the Streamlit UI
streamlit run streamlit_app.py

## Dependencies

pypdf
sentence-transformers
faiss-cpu
numpy
streamlit
python-dotenv
torch
scikit-learn
scipy

## Future Work

Cross-modal reranking using vision–text embeddings
Hybrid retrieval combining dense and lexical search
Retrieval fine-tuning using contrastive learning
Evaluation dashboard for retrieval metrics and latency

Summarization and briefing generation features
