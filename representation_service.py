from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from data_service import preprocess_text # استيراد تابع المعالجة من الخدمة الأولى

def apply_tfidf(documents):
    """
    تابع يأخذ قائمة من النصوص، يقوم بمعالجتها، ثم يمثلها باستخدام TF-IDF
    """
    print("Mathematical evaluation results for this query:")
    # معالجة النصوص أولاً باستخدام التابع الذي بنيناه سابقاً
    processed_docs = [preprocess_text(doc) for doc in documents]
    
    print("Calculating TF-IDF weights...")
    # تهيئة نموذج TF-IDF
    vectorizer = TfidfVectorizer()
    
    # تحويل النصوص المعالجة إلى مصفوفة أرقام
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    
    # الحصول على أسماء الكلمات (المصطلحات)
    feature_names = vectorizer.get_feature_names_out()
    
    # تحويل النتيجة إلى جدول (DataFrame) لسهولة القراءة
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    
    return df

if __name__ == "__main__":
    # عينة نصوص تجريبية بسيطة للتأكد من عمل الخدمة
    sample_docs = [
        "The Manhattan Project and its atomic bomb helped bring an end to World War II.",
        "The success of this project would forever change the world.",
        "Scientific minds were equally important to the success of the Manhattan Project."
    ]
    
    result_df = apply_tfidf(sample_docs)
    
    print("\nMathematical evaluation results for this query:")
    print("=" * 60)
    # طباعة كلمات مختارة فقط كمثال لكي لا يمتلئ الشاشة
    print(result_df[['manhattan', 'project', 'success', 'world', 'bomb']])