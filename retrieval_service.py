from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_service import preprocess_text

def process_and_match_query(query, documents):
    """
    تابع يقوم بمعالجة الاستعلام، تمثيله رياضياً، وحساب التشابه مع الوثائق
    """
    print(f"original query: '{query}'")
    
    # 1. معالجة الاستعلام بنفس طريقة معالجة الوثائق (شرط أساسي لضمان التوافق)
    processed_query = preprocess_text(query)
    print(f"Mathematical evaluation results for this query:")
    print(f"original query: '{query}'")
    print(f"processed query: '{processed_query}'\n")

    # تجهيز الوثائق وتمثيلها (محاكاة لاستدعاء خدمة التمثيل)
    processed_docs = [preprocess_text(doc) for doc in documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    
    query_vector = vectorizer.transform([processed_query])
    
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    results = []
    for doc_id, score in enumerate(similarities):
        if score > 0: # تجاهل الوثائق التي لا تشارك أي كلمة مع الاستعلام
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
    
    print("=" * 60)
    ranked_results = process_and_match_query(user_query, sample_docs)
    
    print("Mathematical evaluation results for this query:")
    print("-" * 60)
    for rank, (doc_id, score, text) in enumerate(ranked_results, 1):
        # طباعة النسبة المئوية للتطابق
        match_percentage = round(score * 100, 1)
        print(f"{rank} Document number{doc_id} | Match: {match_percentage}%")
        print(f"   النص: {text}\n")