from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_service import preprocess_text


def build_tfidf_index(documents):
    """Fit the TF-IDF vectorizer once and reuse the sparse matrix for fast search."""
    print("Building TF-IDF index (once)...")
    processed_docs = [preprocess_text(doc) for doc in documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    return vectorizer, tfidf_matrix


def search_tfidf(query, vectorizer, tfidf_matrix, documents, candidate_doc_ids=None, top_k=100):
    """Search against a prebuilt TF-IDF matrix without rebuilding the index."""
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    results = []
    for doc_id, score in enumerate(similarities):
        if score <= 0:
            continue
        if candidate_doc_ids is not None and doc_id not in candidate_doc_ids:
            continue
        results.append((doc_id, float(score), documents[doc_id]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def process_and_match_query(query, documents):
    """Backward-compatible demo entrypoint that builds the index once per call."""
    print(f"original query: '{query}'")
    processed_query = preprocess_text(query)
    print("Mathematical evaluation results for this query:")
    print(f"original query: '{query}'")
    print(f"processed query: '{processed_query}'\n")

    vectorizer, tfidf_matrix = build_tfidf_index(documents)
    return search_tfidf(processed_query, vectorizer, tfidf_matrix, documents)


if __name__ == "__main__":
    sample_docs = [
        "The Manhattan Project and its atomic bomb helped bring an end to World War II.",
        "The success of this project would forever change the world.",
        "Scientific minds were equally important to the success of the Manhattan Project."
    ]

    user_query = "success of manhattan project"

    print("=" * 60)
    ranked_results = process_and_match_query(user_query, sample_docs)

    print("Mathematical evaluation results for this query:")
    print("-" * 60)
    for rank, (doc_id, score, text) in enumerate(ranked_results, 1):
        match_percentage = round(score * 100, 1)
        print(f"{rank} Document number{doc_id} | Match: {match_percentage}%")
        print(f"   النص: {text}\n")