from spellchecker import SpellChecker
import nltk
from nltk.corpus import wordnet

# تحميل بيانات قاموس المرادفات (يحدث مرة واحدة)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Mathematical evaluation results for this query:")
   # nltk.download('wordnet')
    #nltk.download('omw-1.4')

# تهيئة المصحح الإملائي
spell = SpellChecker()

def correct_spelling(query):
   
    words = query.split()
    corrected_words = []
    
    for word in words:
        # البحث عن التصحيح، وإذا لم يوجد نترك الكلمة كما هي
        correction = spell.correction(word)
        if correction:
            corrected_words.append(correction)
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)

def expand_query(query):
    """
     Mathematical evaluation results for this query:   
    
      """
    words = query.split()
    expanded_words = list(words) 
       
    for word in words:
        # البحث عن مرادف مختلف للكلمة
        added_synonym = False
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                syn_name = lemma.name().replace('_', ' ')
                # إذا وجدنا مرادفاً مختلفاً ولم نضفه مسبقاً
                if syn_name.lower() != word.lower() and syn_name.lower() not in [w.lower() for w in expanded_words]:
                    expanded_words.append(syn_name.lower())
                    added_synonym = True
                    break # يكفي إضافة مرادف واحد للكلمة
            if added_synonym:
                break
                
    return " ".join(expanded_words)

def refine_query(query, apply_spelling=True, apply_expansion=True):
    """
    الدالة الرئيسية التي تدير عملية التحسين بالكامل
    """
    original_query = query
    refined_query = query.lower()
    
    if apply_spelling:
        refined_query = correct_spelling(refined_query)
        
    if apply_expansion:
        refined_query = expand_query(refined_query)
        
    return {
        "original": original_query,
        "refined": refined_query,
        "was_modified": original_query.lower() != refined_query.lower()
    }

if __name__ == "__main__":
    # --- تجربة النظام ---
    print("\n Mathematical evaluation results for this query:")
    test_1 = "succes of manhaten project"
    print(f"original query {test_1}")
    print(f"after refinement: {refine_query(test_1, apply_spelling=True, apply_expansion=False)['refined']}")
    
    print("\n--- Synonym Addition Test ---")
    test_2 = "fast car"
    print(f"original query {test_2}")
    print(f"after refinement: {refine_query(test_2, apply_spelling=False, apply_expansion=True)['refined']}")