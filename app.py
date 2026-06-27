import streamlit as st
import requests
import time
import pandas as pd # أضفنا هذه المكتبة لرسم جداول التقييم

st.set_page_config(page_title="Smart Search Engine", page_icon="🔍", layout="wide")

# --- إدارة حالة الواجهة (Session State) ---
if 'limit' not in st.session_state:
    st.session_state.limit = 10
if 'execute_search' not in st.session_state:
    st.session_state.execute_search = False

st.markdown("""
    <style>
    .result-card { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .score-badge { background-color: #e1f5fe; color: #0277bd; padding: 4px 10px; border-radius: 15px; font-size: 13px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3176/3176298.png", width=120) 
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    dataset_map = {"Lotte (General Text)": "lotte", "Quora (Questions and Answers)": "quora"}    
    display_dataset = st.selectbox("Choose the dataset:", list(dataset_map.keys()), label_visibility="collapsed")
    api_dataset_choice = dataset_map[display_dataset]

    algorithms_map = {"TF-IDF ": "tfidf", "BM25 ": "bm25", "BERT ": "bert", "Hybrid Parallel": "hybrid_parallel", "Hybrid Serial": "hybrid_serial"}
    display_choice = st.radio("Choose the algorithm:", list(algorithms_map.keys()), label_visibility="collapsed")
    api_algo_choice = algorithms_map[display_choice]

    k1_val, b_val, alpha_val = 1.5, 0.75, 0.5
    with st.expander("🛠️ Advanced Parameter Settings"):
        if api_algo_choice == "bm25":
            k1_val = st.slider("k1 coefficient", 0.1, 3.0, 1.5, 0.1)
            b_val = st.slider("b coefficient", 0.0, 1.0, 0.75, 0.05)
        elif api_algo_choice == "hybrid_parallel":
            alpha_val = st.slider("Weight of BM25 vs. BERT", 0.0, 1.0, 0.5, 0.1)

    st.markdown("---")
    use_refinement = st.toggle("✨ Enable AI (Correction + Synonyms)", value=False)

st.title("🔍 Smart Search Engine (IR 2026)")

tab_search, tab_analytics = st.tabs(["🚀 Search Interface", "📊 Academic Evaluation Dashboard"])


with tab_search:
    search_col, btn_col = st.columns([5, 1])
    with search_col:
        query = st.text_input("Search", placeholder="What are you looking for?", label_visibility="collapsed")
    with btn_col:
        search_button = st.button("search 🚀", use_container_width=True, type="primary")

    if search_button:
        if query:
            st.session_state.limit = 10
            st.session_state.execute_search = True
        else:
            st.warning("⚠️ Please enter search terms first.")

    if st.session_state.execute_search and query:
        api_url = "http://127.0.0.1:8009/search" 
        payload = {
            "query": query, "dataset": api_dataset_choice, "algorithm": api_algo_choice,
            "k1": k1_val, "b": b_val, "alpha": alpha_val, "use_refinement": use_refinement,
            "limit": st.session_state.limit
        }
        
        start_time = time.time()
        with st.spinner('⏳ Searching in the data...'):
            try:
                response = requests.post(api_url, json=payload)
                search_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("refinement_info") and data["refinement_info"]["was_modified"]:
                        st.info(f"💡 **Improvement in Query:** The search was performed for ` {data['final_query_used']} ` instead of ` {query} `.")
                    
                    st.markdown("---")
                    met1, met2, met3 = st.columns(3)
                    met1.metric("Total Results Found", f"{data['total_hits']} Documents")
                    met2.metric("Response Time", f"{search_time:.3f} seconds")
                    met3.metric("Currently Displayed", f"{len(data['results'])} Results")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    for index, res in enumerate(data['results'], 1):
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h4 style="margin: 0; color: #333;">#{index} | Document ID: {res['doc_id']}</h4>
                                <span class="score-badge">🎯 Score: {res['score']}</span>
                            </div>
                            <p style="margin: 0; font-size: 16px; color: #555;">{res['text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if len(data['results']) < data['total_hits']:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Display 10 Additional Results ⬇️", use_container_width=True):
                            st.session_state.limit += 10
                            st.rerun()
                            
                else:
                    st.error("❌ Sorry, an error occurred on the server.")
            except Exception as e:
                st.error("❌ Connection Error! Please ensure the uvicorn server is running.")
