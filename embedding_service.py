from sentence_transformers import SentenceTransformer, util

model = None


def get_model():
    global model
    if model is None:
        print("Loading BERT model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def build_embedding_index(documents):
    print("Converting documents to embeddings... (This may take some time for large datasets)")
    document_embeddings = get_model().encode(documents, convert_to_tensor=True)
    return document_embeddings


def search_embeddings(query, document_embeddings, documents, top_k=100):
    """
    performs search for the query using cosine similarity
    """
    query_embedding = get_model().encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]

    results = []
    for doc_id, score in enumerate(cosine_scores):
        results.append((doc_id, score.item(), documents[doc_id]))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    # عينة لتجربة عمل النموذج محلياً
    sample_docs = [
        "The Manhattan Project and its atomic bomb helped bring an end to World War II.",
        "The success of this project would forever change the world.",
        "Scientific minds were equally important to the success of the Manhattan Project."
    ]
    
    user_query = "success of manhattan project"
    
    # التجربة
    doc_embs = build_embedding_index(sample_docs)
    res = search_embeddings(user_query, doc_embs, sample_docs)
    
    print("\n--- Search Results for BERT Embeddings ---")
    for rank, (doc_id, score, text) in enumerate(res, 1):
        # تحويل الرقم إلى نسبة مئوية للسهولة
        print(f"{rank}. Document: {doc_id} | Similarity: {round(score * 100, 1)}%")
        print(f"   Text: {text}\n")