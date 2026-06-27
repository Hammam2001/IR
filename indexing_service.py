from data_service import preprocess_text
from collections import defaultdict

def build_inverted_index(documents):
    """
    Building an Inverted Index from a list of documents.
    The index maps each word to a list of document IDs where it appears.
    """
    print("Building the inverted index...")
    
    # Using defaultdict to facilitate adding document IDs to words
    inverted_index = defaultdict(list)
    
    for doc_id, text in enumerate(documents):
        # 1. معالجة النص أولاً باستخدام الخدمة الأولى
        processed_text = preprocess_text(text)
        
        # 2. تقسيم النص إلى كلمات
        words = processed_text.split()
        
        # 3. إضافة رقم المستند إلى قائمة كل كلمة ظهرت فيه 
        # نستخدم set(words) لضمان عدم تكرار رقم المستند إذا تكررت الكلمة داخله
        for word in set(words):
            inverted_index[word].append(doc_id)
            
    print("Index built successfully!")
    return dict(inverted_index)

if __name__ == "__main__":
    # نصوص تجريبية للتأكد من عمل الخدمة
    sample_docs = [
        "The Manhattan Project and its atomic bomb helped bring an end to World War II.",
        "The success of this project would forever change the world.",
        "Scientific minds were equally important to the success of the Manhattan Project."
    ]
    
    index = build_inverted_index(sample_docs)
    
    print("\nSample from the inverted index:")
    print("=" * 60)
    
    # البحث عن كلمات معينة لمعرفة المستندات التي تحتويها
    words_to_check = ['manhattan', 'project', 'success', 'scientific']
    for word in words_to_check:
        print(f"word '{word}' appears in documents with IDs: {index.get(word, [])}")