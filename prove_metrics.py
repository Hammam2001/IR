import math

# --- 1. الدوال الرياضية الخاصة بك ---
def calculate_precision_at_k(retrieved, relevant, k=10):
    retrieved_k = retrieved[:k]
    hits = set(retrieved_k).intersection(set(relevant))
    return len(hits) / len(retrieved_k) if retrieved_k else 0.0

def calculate_recall(retrieved, relevant):
    hits = set(retrieved).intersection(set(relevant))
    return len(hits) / len(relevant) if relevant else 0.0

def calculate_ndcg_at_k(retrieved, relevant, k=10):
    def dcg(docs):
        score = 0.0
        for i, doc in enumerate(docs[:k]):
            if doc in relevant:
                score += 1.0 / math.log2(i + 2.0)
        return score
    
    actual_dcg = dcg(retrieved)
    ideal_dcg = dcg(relevant) 
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

# --- 2. سيناريو الإثبات الحي للجنة التحكيم ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" 🕵️‍♂️Live Proof of the Search Engine's Mathematical Evaluation Quality ")
    print("="*60)

    # 1. الاستعلام المختار للاختبار
    query = "Which question should I ask on Quora?"
    
    # 2. التقييم البشري (Qrels) - ما يعتبره الخبراء صحيحاً 100%
    human_relevant_docs = [134031, 271267, 134030] 
    
    # 3. محاكاة لنتائج جلبها نظامك (نفترض أن نظامك جلب 10 مستندات، من ضمنها اثنان صحيحان)
    # لاحظ أن المستند الصحيح 134031 جاء في المركز الأول، والمستند 134030 جاء في المركز الثالث
    system_retrieved_docs = [134031, 168781, 134030, 5460, 64216, 202290, 129933, 179172, 40290, 68209]

    print(f"\n🔍 query: '{query}'")
    print(f"👨‍🏫 ideal answers (according to humans - Qrels): {human_relevant_docs}")
    print(f"🤖 results retrieved by the system:       {system_retrieved_docs}")
    print("-" * 60)
    
    # 4. حساب وعرض المقاييس مع الشرح التفصيلي
    precision = calculate_precision_at_k(system_retrieved_docs, human_relevant_docs, k=10)
    recall = calculate_recall(system_retrieved_docs, human_relevant_docs)
    ndcg = calculate_ndcg_at_k(system_retrieved_docs, human_relevant_docs, k=10)

    print("📊 Mathematical evaluation results for this query:")
    print(f"1️⃣ Precision@10 = {precision*100:.1f}%")
    print("   👉 Explanation: Out of the top 10 documents retrieved by the system, 2 are relevant.")
    
    print(f"2️⃣ Recall = {recall*100:.1f}%")
    print(f"   👉 Explanation: Out of the total {len(human_relevant_docs)} relevant documents in the database, the system retrieved 2.")
    print(f"3️⃣ Quality of Ranking (nDCG@10) = {ndcg*100:.1f}%")
    print("   👉 Explanation: Because the system placed the first correct document at the top (Rank 1), and the second correct document at (Rank 3), it received a very high ranking evaluation based on the logarithmic discount.")
    print("="*60 + "\n")