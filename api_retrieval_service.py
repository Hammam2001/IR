import os
os.environ["IR_DATASETS_HOME"] = "E:/ir_datasets_data"

from fastapi import FastAPI
from pydantic import BaseModel
import ir_datasets
import pickle

from retrieval_service import build_tfidf_index, search_tfidf
from bm25_service import build_bm25_model, search_bm25
from embedding_service import build_embedding_index, search_embeddings
from vector_store_service import build_faiss_from_embeddings, search_faiss
from hybrid_service import parallel_hybrid_search, serial_hybrid_search
from query_refinement_service import refine_query
from indexing_service import build_inverted_index
from data_service import preprocess_text

app = FastAPI(title="Sustainable Search and Retrieval Service")


class SearchQuery(BaseModel):
    query: str
    dataset: str = "quora"
    algorithm: str = "tfidf"
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.5
    use_refinement: bool = False
    limit: int = 10


datasets_store = {}


def _build_dataset_payload(docs, doc_ids, embeddings=None):
    if embeddings is None:
        embeddings = build_embedding_index(docs)

    bm25_model = build_bm25_model(docs)
    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(docs)
    inverted_index = build_inverted_index(docs)
    faiss_index = build_faiss_from_embeddings(embeddings)

    return {
        "docs": docs,
        "doc_ids": doc_ids,
        "bm25": bm25_model,
        "embeddings": embeddings,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "inverted_index": inverted_index,
        "bm25_models": {(1.5, 0.75): bm25_model},
        "faiss_index": faiss_index,
    }


def _get_candidate_doc_ids(query, inverted_index, documents):
    tokens = preprocess_text(query).split()
    if not tokens or not inverted_index:
        return None

    candidate_ids = set()
    for token in tokens:
        candidate_ids.update(inverted_index.get(token, []))

    if not candidate_ids:
        return None
    return sorted(candidate_ids)


def load_and_index_dataset(dataset_name, ir_name):
    cache_file = f"{dataset_name}_cache_full.pkl"

    if os.path.exists(cache_file):
        print(f"\n📂 We found a saved file for...{dataset_name} ...on the hard drive! Loading instantly...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        docs = cached_data.get("docs", [])
        doc_ids = cached_data.get("doc_ids") or [str(i) for i in range(len(docs))]
        embeddings = cached_data.get("embeddings")

        if not docs:
            dataset = ir_datasets.load(ir_name)
            docs = [doc.text for doc in dataset.docs_iter()]
            doc_ids = [getattr(doc, "doc_id", str(i)) for i, doc in enumerate(dataset.docs_iter())]

        payload = {
            "docs": docs,
            "doc_ids": doc_ids,
            "bm25": cached_data.get("bm25") or build_bm25_model(docs),
            "embeddings": embeddings if embeddings is not None else build_embedding_index(docs),
            "tfidf_vectorizer": cached_data.get("tfidf_vectorizer"),
            "tfidf_matrix": cached_data.get("tfidf_matrix"),
            "inverted_index": cached_data.get("inverted_index") or build_inverted_index(docs),
            "bm25_models": cached_data.get("bm25_models", {}),
            "faiss_index": None,
        }
        if payload["tfidf_vectorizer"] is None or payload["tfidf_matrix"] is None:
            payload["tfidf_vectorizer"], payload["tfidf_matrix"] = build_tfidf_index(docs)
        payload["bm25_models"].setdefault((1.5, 0.75), payload["bm25"])
        payload["faiss_index"] = build_faiss_from_embeddings(payload["embeddings"])
        datasets_store[dataset_name] = payload
        print(f"✅ Successfully loaded {dataset_name} (complete) from the hard drive!")
        return

    print(f"\n⚠️ Missing Cache: We didn't find a saved file for {dataset_name}.")
    print(f"⏳ Loading and indexing the entire dataset (without any truncation)... This might take some time!")

    dataset = ir_datasets.load(ir_name)
    docs = []
    doc_ids = []
    for doc in dataset.docs_iter():
        docs.append(doc.text)
        doc_ids.append(getattr(doc, "doc_id", str(len(doc_ids))))

    print(f"✅ Successfully loaded {len(docs)} documents completely. Building models for {dataset_name}...")
    datasets_store[dataset_name] = _build_dataset_payload(docs, doc_ids)

    print(f"💾 Saving data and indices...{dataset_name} on the hard drive for sustainability...")
    cache_payload = dict(datasets_store[dataset_name])
    cache_payload.pop("faiss_index", None)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_payload, f)
    print(f"✅ Successfully saved {dataset_name} as a secure file ({cache_file}).")


load_and_index_dataset("lotte", "lotte/lifestyle/dev/search")
load_and_index_dataset("quora", "beir/quora")

print("\n🚀 [Sustainable System] All (complete) data successfully prepared! The server is fully ready.")
# -----------------------------------------------------------------


@app.post("/search")
def search_api(request: SearchQuery):
    formatted_results = []

    selected_data = datasets_store.get(request.dataset)
    if not selected_data:
        return {"error": "Dataset not found"}

    real_docs = selected_data["docs"]
    doc_ids = selected_data.get("doc_ids", [str(i) for i in range(len(real_docs))])
    global_bm25_model = selected_data["bm25"]
    global_embeddings = selected_data["embeddings"]
    final_query_text = request.query
    refinement_info = None
    algorithm = request.algorithm.lower().strip()

    if request.use_refinement:
        refinement_data = refine_query(request.query)
        final_query_text = refinement_data["refined"]
        refinement_info = refinement_data

    candidate_doc_ids = _get_candidate_doc_ids(final_query_text, selected_data.get("inverted_index"), real_docs)

    if algorithm == "tfidf":
        raw_results = search_tfidf(
            final_query_text,
            selected_data["tfidf_vectorizer"],
            selected_data["tfidf_matrix"],
            real_docs,
            candidate_doc_ids=candidate_doc_ids,
        )
    elif algorithm == "bm25":
        if request.k1 != 1.5 or request.b != 0.75:
            cache_key = (round(request.k1, 3), round(request.b, 3))
            selected_data.setdefault("bm25_models", {})
            custom_bm25 = selected_data["bm25_models"].get(cache_key)
            if custom_bm25 is None:
                custom_bm25 = build_bm25_model(real_docs, k1=request.k1, b=request.b)
                selected_data["bm25_models"][cache_key] = custom_bm25
            raw_results = search_bm25(final_query_text, custom_bm25, real_docs, candidate_doc_ids=candidate_doc_ids)
        else:
            raw_results = search_bm25(final_query_text, global_bm25_model, real_docs, candidate_doc_ids=candidate_doc_ids)
    elif algorithm == "bert":
        if selected_data.get("faiss_index") is not None:
            raw_results = search_faiss(final_query_text, selected_data["faiss_index"], real_docs)
        else:
            raw_results = search_embeddings(final_query_text, global_embeddings, real_docs)
    elif algorithm == "hybrid_parallel":
        raw_results = parallel_hybrid_search(final_query_text, global_bm25_model, global_embeddings, real_docs, alpha=request.alpha)
    elif algorithm == "hybrid_serial":
        raw_results = serial_hybrid_search(final_query_text, global_bm25_model, global_embeddings, real_docs, initial_k=100)
    else:
        return {
            "original_query": request.query,
            "final_query_used": final_query_text,
            "refinement_info": refinement_info,
            "dataset_used": request.dataset,
            "algorithm_used": request.algorithm,
            "total_hits": 0,
            "results_count": 0,
            "results": [],
            "error": f"Unsupported algorithm: {request.algorithm}",
        }

    total_hits = len(raw_results)

    for doc_index, score, text in raw_results[:request.limit]:
        actual_doc_id = doc_ids[doc_index] if doc_index < len(doc_ids) else doc_index
        display_score = f"{round(score * 100, 1)}%" if algorithm == "tfidf" else str(round(score, 3))
        formatted_results.append({
            "doc_id": actual_doc_id,
            "score": display_score,
            "text": text
        })

    return {
        "original_query": request.query,
        "final_query_used": final_query_text,
        "refinement_info": refinement_info,
        "dataset_used": request.dataset,
        "algorithm_used": algorithm,
        "total_hits": total_hits,
        "results_count": len(formatted_results),
        "results": formatted_results
    }