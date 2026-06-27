from bm25_service import build_bm25_model, search_bm25
from embedding_service import build_embedding_index, search_embeddings, get_model
from sentence_transformers import util

def min_max_normalize(scores):
    """Helper function to normalize the weights to be between 0 and 1 so they can be combined."""
    min_val = min(scores)
    max_val = max(scores)
    if max_val == min_val:
        return [0.0 for _ in scores]
    return [(s - min_val) / (max_val - min_val) for s in scores]

def parallel_hybrid_search(query, bm25_model, document_embeddings, documents, alpha=0.5):
    """
    Parallel Hybrid Search
    alpha: weight ratio for BM25 (default 0.5 means 50% BM25 and 50% BERT)
    """
    # 1. Fetch BM25 weights for all documents
    bm25_results = search_bm25(query, bm25_model, documents)
    bm25_scores = [0] * len(documents)
    for doc_id, score, _ in bm25_results:
        bm25_scores[doc_id] = score
        
    # 2. جلب أوزان BERT لكل المستندات
    query_emb = get_model().encode(query, convert_to_tensor=True)
    bert_scores_tensor = util.cos_sim(query_emb, document_embeddings)[0]
    bert_scores = [s.item() for s in bert_scores_tensor]
    
    # 3. توحيد الأوزان (Normalization)
    norm_bm25 = min_max_normalize(bm25_scores)
    norm_bert = min_max_normalize(bert_scores)
    
    # 4. دمج الأوزان (Scoring Fusion)
    hybrid_results = []
    for doc_id in range(len(documents)):
        final_score = (alpha * norm_bm25[doc_id]) + ((1 - alpha) * norm_bert[doc_id])
        hybrid_results.append((doc_id, final_score, documents[doc_id]))
        
    # ترتيب وإعادة النتائج كاملة دون اقتطاع
    hybrid_results = sorted(hybrid_results, key=lambda x: x[1], reverse=True)
    return hybrid_results[:100]

def serial_hybrid_search(query, bm25_model, document_embeddings, documents, initial_k=100):
    """
    Serial Hybrid Search
    BM25 is used first to retrieve the top initial_k documents, then BERT re-ranks them
    """
    bm25_results = search_bm25(query, bm25_model, documents)
    top_bm25_docs = bm25_results[:initial_k]
    
    candidate_doc_ids = [doc_id for doc_id, _, _ in top_bm25_docs]
    
    query_emb = get_model().encode(query, convert_to_tensor=True)
    
    serial_results = []
    for doc_id in candidate_doc_ids:
        doc_emb = document_embeddings[doc_id]
        score = util.cos_sim(query_emb, doc_emb)[0][0].item()
        serial_results.append((doc_id, score, documents[doc_id]))
        
    serial_results = sorted(serial_results, key=lambda x: x[1], reverse=True)
    
    return serial_results

if __name__ == "__main__":
    sample_docs = [
        "The Manhattan Project and its atomic bomb helped bring an end to World War II.",
        "The success of this project would forever change the world.",
        "Scientific minds were equally important to the success of the Manhattan Project."
    ]
    
    user_query = "success of manhattan project"
    
    print("Preparing...")
    b_model = build_bm25_model(sample_docs)
    d_embs = build_embedding_index(sample_docs)
    
    print("\n--- Parallel Hybrid ---")
    parallel_res = parallel_hybrid_search(user_query, b_model, d_embs, sample_docs)
    for rank, (d_id, sc, txt) in enumerate(parallel_res, 1):
        print(f"{rank}. Document: {d_id} | Hybrid Score: {round(sc, 3)} | {txt}")
        
    print("\n--- Serial Hybrid ---")
    serial_res = serial_hybrid_search(user_query, b_model, d_embs, sample_docs)
    for rank, (d_id, sc, txt) in enumerate(serial_res, 1):
        print(f"{rank}. Document: {d_id} | Final Score: {round(sc, 3)} | {txt}")