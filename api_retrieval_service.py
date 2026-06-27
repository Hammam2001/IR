import os
os.environ["IR_DATASETS_HOME"] = "E:/ir_datasets_data"



from fastapi import FastAPI
from pydantic import BaseModel
import ir_datasets
import os
import pickle 

from retrieval_service import process_and_match_query 
from bm25_service import build_bm25_model, search_bm25
from embedding_service import build_embedding_index, search_embeddings
from hybrid_service import parallel_hybrid_search, serial_hybrid_search
from query_refinement_service import refine_query

app = FastAPI(title="Sustainable Search and Retrieval Service")

class SearchQuery(BaseModel):
    query: str
    dataset: str = "quora" 
    algorithm: str = "tfidf" 
    k1: float = 1.5
    b: float = 0.75
    alpha: float = 0.5 
    use_refinement: bool = False
    limit: int = 10

datasets_store = {}

def load_and_index_dataset(dataset_name, ir_name):

    cache_file = f"{dataset_name}_cache_full.pkl"
    
    if os.path.exists(cache_file):
        print(f"\n📂 We found a saved file for...{dataset_name} ...on the hard drive! Loading instantly...")
        with open(cache_file, 'rb') as f:
            datasets_store[dataset_name] = pickle.load(f)
        print(f"✅ Successfully loaded {dataset_name} (complete) from the hard drive!")
        return

    print(f"\n⚠️ Missing Cache: We didn't find a saved file for {dataset_name}.")
    print(f"⏳ Loading and indexing the entire dataset (without any truncation)... This might take some time!")
    
    dataset = ir_datasets.load(ir_name)
    docs = []
    
    for doc in dataset.docs_iter():
        docs.append(doc.text)
            
    print(f"✅ Successfully loaded {len(docs)} documents completely. Building models for {dataset_name}...")
    
    bm25_model = build_bm25_model(docs)
    embeddings = build_embedding_index(docs)
    
    datasets_store[dataset_name] = {
        "docs": docs,
        "bm25": bm25_model,
        "embeddings": embeddings
    }
    
    print(f"💾 Saving data and indices...{dataset_name} on the hard drive for sustainability...")
    with open(cache_file, 'wb') as f:
        pickle.dump(datasets_store[dataset_name], f)
    print(f"✅ Successfully saved {dataset_name} as a secure file ({cache_file}).")


load_and_index_dataset("lotte", "lotte/lifestyle/dev/search")
load_and_index_dataset("quora", "beir/quora")

print("\n🚀 [Sustainable System] All (complete) data successfully prepared! The server is fully ready.")
# -----------------------------------------------------------------

@app.post("/search")
def search_api(request: SearchQuery):
    formatted_results = []
    
    selected_data = datasets_store.get(request.dataset)
    if not selected_data:
        return {"error": "Dataset not found"}
        
    real_docs = selected_data["docs"]
    global_bm25_model = selected_data["bm25"]
    global_embeddings = selected_data["embeddings"]
    
    final_query_text = request.query
    refinement_info = None
    if request.use_refinement:
        refinement_data = refine_query(request.query)
        final_query_text = refinement_data["refined"]
        refinement_info = refinement_data
    
    if request.algorithm == "tfidf":
        raw_results = process_and_match_query(final_query_text, real_docs)
    elif request.algorithm == "bm25":
        if request.k1 != 1.5 or request.b != 0.75:
            custom_bm25 = build_bm25_model(real_docs, k1=request.k1, b=request.b)
            raw_results = search_bm25(final_query_text, custom_bm25, real_docs)
        else:
            raw_results = search_bm25(final_query_text, global_bm25_model, real_docs)
    elif request.algorithm == "bert":
        raw_results = search_embeddings(final_query_text, global_embeddings, real_docs)
    elif request.algorithm == "hybrid_parallel":
        raw_results = parallel_hybrid_search(final_query_text, global_bm25_model, global_embeddings, real_docs, alpha=request.alpha)
    elif request.algorithm == "hybrid_serial":
        raw_results = serial_hybrid_search(final_query_text, global_bm25_model, global_embeddings, real_docs, initial_k=50)

    total_hits = len(raw_results)

    for doc_id, score, text in raw_results[:request.limit]:
        display_score = f"{round(score * 100, 1)}%" if request.algorithm == "tfidf" else str(round(score, 3))
        formatted_results.append({
            "doc_id": doc_id,
            "score": display_score,
            "text": text
        })
            
    return {
        "original_query": request.query,
        "final_query_used": final_query_text,
        "refinement_info": refinement_info,
        "dataset_used": request.dataset,
        "algorithm_used": request.algorithm,
        "total_hits": total_hits,               
        "results_count": len(formatted_results), 
        "results": formatted_results
    }