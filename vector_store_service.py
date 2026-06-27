import faiss
import numpy as np
from embedding_service import model # استيراد نموذج BERT الخاص بنا

def build_faiss_vector_store(documents):
    """
    يقوم ببناء قاعدة بيانات متجهات (Vector Store) باستخدام FAISS
    لتسريع عمليات البحث الدلالي بشكل هائل.
    """
    print("Calculating embeddings for documents...")
    # تحويل النصوص إلى متجهات بصيغة Numpy (مطلوبة لـ FAISS)
    embeddings = model.encode(documents, convert_to_numpy=True)
    
    # تحديد عدد الأبعاد (غالباً 384 لنموذجنا)
    dimension = embeddings.shape[1]
    
    # بناء الـ Vector Store (باستخدام مقياس L2 Distance للسرعة الفائقة)
    vector_store = faiss.IndexFlatL2(dimension)
    
    # إضافة المتجهات لقاعدة البيانات
    vector_store.add(embeddings)
    print(f"Calculating embeddings for documents... Done!")
    
    return vector_store, embeddings

def search_vector_store(query, vector_store, documents, top_k=10):
    """
    البحث السريع جداً داخل قاعدة بيانات المتجهات
    """
    # تحويل الاستعلام إلى متجه
    query_vector = model.encode([query], convert_to_numpy=True)
    
    # البحث السريع في FAISS (يعيد المسافات والأرقام التعريفية)
    distances, indices = vector_store.search(query_vector, top_k)
    
    results = []
    for i in range(top_k):
        doc_id = indices[0][i]
        if doc_id != -1: # تأكيد وجود المستند
            # تحويل المسافة (Distance) إلى نسبة تشابه (Score)
            score = 1.0 / (1.0 + distances[0][i])
            results.append((doc_id, float(score), documents[doc_id]))
            
    return results

if __name__ == "__main__":
    # تجربة الميزة الإضافية
    sample_docs = [
        "The Manhattan Project and its atomic bomb helped bring an end to World War II.",
        "The success of this project would forever change the world.",
        "Scientific minds were equally important to the success of the Manhattan Project."
    ]
    
    print("--- Synonym Addition Test (Vector Store) ---")
    store, _ = build_faiss_vector_store(sample_docs)
    
    q = "success of manhattan project"
    results = search_vector_store(q, store, sample_docs)
    
    print("\nMathematical evaluation results for this query:")
    for rank, (doc_id, score, text) in enumerate(results, 1):
        print(f"{rank} Document number{doc_id} | Weight: {round(score, 3)} | {text}")