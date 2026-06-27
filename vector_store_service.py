import faiss
import numpy as np
from embedding_service import get_model


def build_faiss_from_embeddings(embeddings):
    """Build a cosine FAISS index from already-computed BERT embeddings."""
    emb = embeddings.cpu().numpy() if hasattr(embeddings, "cpu") else np.asarray(embeddings)
    emb = np.ascontiguousarray(emb, dtype="float32")
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def build_faiss_vector_store(documents):
    """Backward-compatible helper that builds a FAISS index from document text."""
    print("Calculating embeddings for documents...")
    embeddings = get_model().encode(documents, convert_to_numpy=True)
    index = build_faiss_from_embeddings(embeddings)
    print("Calculating embeddings for documents... Done!")
    return index, embeddings


def search_faiss(query, index, documents, top_k=100):
    """Search the existing FAISS index using cosine similarity."""
    query_vector = get_model().encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_k)

    results = []
    for i in range(len(indices[0])):
        doc_id = int(indices[0][i])
        if doc_id != -1:
            results.append((doc_id, float(scores[0][i]), documents[doc_id]))
    return results


if __name__ == "__main__":
    sample_docs = [
        "The Manhattan Project and its atomic bomb helped bring an end to World War II.",
        "The success of this project would forever change the world.",
        "Scientific minds were equally important to the success of the Manhattan Project."
    ]

    print("--- Synonym Addition Test (Vector Store) ---")
    store, _ = build_faiss_vector_store(sample_docs)

    q = "success of manhattan project"
    results = search_faiss(q, store, sample_docs)

    print("\nMathematical evaluation results for this query:")
    for rank, (doc_id, score, text) in enumerate(results, 1):
        print(f"{rank} Document number{doc_id} | Weight: {round(score, 3)} | {text}")