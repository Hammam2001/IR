from sentence_transformers import SentenceTransformer, util

# تحميل نموذج خفيف وسريع مناسب لعمليات البحث
print("Loading BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_embedding_index(documents):
   
    print("Converting documents to embeddings... (This may take some time for large datasets)")
    # تحويل النصوص إلى متجهات (Tensors)
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    return document_embeddings

def search_embeddings(query, document_embeddings, documents, top_k=100):
    """
    performs search for the query using cosine similarity
    """
    # تحويل الاستعلام إلى متجه بنفس الطريقة
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # حساب التشابه الجيبي (Cosine Similarity) بين متجه الاستعلام ومتجهات كل المستندات
    # يعيد هذا التابع مصفوفة من الأوزان (Scores)
    cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]
    
    results = []
    for doc_id, score in enumerate(cosine_scores):
        # تحويل القيمة من Tensor إلى رقم عادي
        results.append((doc_id, score.item(), documents[doc_id]))
        
    # ترتيب النتائج من الأعلى تشابهاً للأقل
    results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # إعادة أفضل K نتائج فقط
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