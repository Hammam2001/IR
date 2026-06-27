# Smart Search Engine — Information Retrieval 2026

A custom search engine over two large datasets (Quora and Lotte) supporting five retrieval models — TF-IDF, BM25, BERT embeddings, and two hybrids — exposed through a FastAPI service with a Streamlit UI.

## Architecture
UI (Streamlit) → REST API (FastAPI) → retrieval services → ranked results.
Indexes such as BM25, TF-IDF, and FAISS are built once at startup and reused for later queries.

## Datasets
- beir/quora
- lotte/lifestyle/dev/search

## Requirements
Python 3.10+ is recommended. Install dependencies with:

```bash
pip install fastapi uvicorn streamlit ir_datasets rank_bm25 sentence-transformers scikit-learn nltk faiss-cpu pyspellchecker pandas requests
```

## Run the backend
```bash
uvicorn api_retrieval_service:app --port 8009
```

## Run the UI
In a second terminal:

```bash
streamlit run app.py
```

## Evaluation
With the backend running:

```bash
python evaluate_all.py
python prove_metrics.py
```

## Project structure
- data_service.py: preprocessing
- indexing_service.py: inverted index
- retrieval_service.py: TF-IDF index and search
- bm25_service.py: BM25 search
- embedding_service.py: BERT embeddings
- vector_store_service.py: FAISS vector store
- hybrid_service.py: hybrid retrieval
- query_refinement_service.py: query refinement
- api_retrieval_service.py: FastAPI gateway
- app.py: Streamlit UI
