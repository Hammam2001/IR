from rank_bm25 import BM25Okapi
from data_service import preprocess_text

BM25_CACHE = {}


def build_bm25_model(documents, k1=1.5, b=0.75):
    cache_key = (id(documents), round(k1, 3), round(b, 3))
    if cache_key in BM25_CACHE:
        return BM25_CACHE[cache_key]

    print(f"Building BM25 (k1={k1}, b={b})...")
    tokenized_docs = [preprocess_text(doc).split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    BM25_CACHE[cache_key] = bm25
    return bm25


def search_bm25(query, bm25_model, documents, candidate_doc_ids=None):
    tokenized_query = preprocess_text(query).split()
    doc_scores = bm25_model.get_scores(tokenized_query)

    results = []
    for doc_id, score in enumerate(doc_scores):
        if score <= 0:
            continue
        if candidate_doc_ids is not None and doc_id not in candidate_doc_ids:
            continue
        results.append((doc_id, score, documents[doc_id]))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results


if __name__ == "__main__":
    sample_docs = [
        "The Manhattan Project and its atomic bomb helped bring an end to World War II.",
        "The success of this project would forever change the world.",
        "Scientific minds were equally important to the success of the Manhattan Project."
    ]

    user_query = "success of manhattan project"

    model = build_bm25_model(sample_docs, k1=1.5, b=0.75)
    results = search_bm25(user_query, model, sample_docs)

    print("\n--- Search Results for BM25 Model ---")
    for rank, (doc_id, score, text) in enumerate(results, 1):
        print(f"{rank}. Document: {doc_id} | Score: {round(score, 3)}")
        print(f"   Text: {text}\n")