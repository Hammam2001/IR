import math


def calculate_precision_at_k(retrieved_docs, relevant_docs, k=10):
    """Calculating metric...Precision@K"""
    retrieved_k = retrieved_docs[:k]
    relevant_and_retrieved = set(retrieved_k).intersection(set(relevant_docs))
    if not retrieved_k:
        return 0.0
    return len(relevant_and_retrieved) / len(retrieved_k)

def calculate_recall(retrieved_docs, relevant_docs):
    """Calculating metric...Recall"""
    if not relevant_docs:
        return 0.0
    relevant_and_retrieved = set(retrieved_docs).intersection(set(relevant_docs))
    return len(relevant_and_retrieved) / len(relevant_docs)

def calculate_average_precision(retrieved_docs, relevant_docs):
    """Calculating metric...Average Precision (AP)"""
    if not relevant_docs:
        return 0.0
    score = 0.0
    num_hits = 0.0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / len(relevant_docs)

def calculate_ndcg_at_k(retrieved_docs, relevant_docs, k=10):
    """Calculating metric...nDCG (Normalized Discounted Cumulative Gain)"""
    def dcg(docs):
        score = 0.0
        for i, doc in enumerate(docs[:k]):
            if doc in relevant_docs:
                score += 1.0 / math.log2(i + 2.0)
        return score
    
    actual_dcg = dcg(retrieved_docs)
    ideal_dcg = dcg(relevant_docs) 
    
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


if __name__ == "__main__":
    actual_relevant_docs = [2, 4, 5, 8]
    
    models_results = {
        "TF-IDF": [1, 2, 3, 4, 11, 6, 7, 8, 9, 10],            
        "BM25": [2, 1, 8, 3, 5, 4, 7, 6, 9, 10],              
        "BERT (Vector Store)": [4, 2, 8, 12, 1, 3, 5, 6, 7, 9], 
        "Hybrid (Parallel)": [2, 4, 5, 8, 15, 1, 3, 11, 6, 7] 
    }

    print("Calculating comprehensive evaluation metrics for the system...")
    print("=" * 75)
    print(f"{'الخوارزمية':<22} | {'P@10':<10} | {'Recall':<10} | {'MAP':<10} | {'nDCG@10':<10}")
    print("-" * 75)

    for model_name, system_retrieved_docs in models_results.items():
        precision = calculate_precision_at_k(system_retrieved_docs, actual_relevant_docs, k=10)
        recall = calculate_recall(system_retrieved_docs, actual_relevant_docs)
        ap = calculate_average_precision(system_retrieved_docs, actual_relevant_docs)
        ndcg = calculate_ndcg_at_k(system_retrieved_docs, actual_relevant_docs, k=10)
        
        print(f"{model_name:<25} | {precision*100:<9.1f}% | {recall*100:<9.1f}% | {ap*100:<9.1f}% | {ndcg*100:<9.1f}%")
        
    print("=" * 75)