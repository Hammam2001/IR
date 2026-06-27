import ir_datasets

print("Loading Quora test data...")
# نستخدم مجموعة quora لأنك قمت بتحميلها بالكامل مسبقاً وتحتوي على الأسئلة (Queries) والتقييمات (Qrels)
dataset = ir_datasets.load("beir/quora/test")

print("\n--- 📝 Examples of Queries ---")
count = 0
for query in dataset.queries_iter():
    print(f"Query ID: {query.query_id} | Text: {query.text}")
    count += 1
    if count >= 5: # Printing only 5 examples
        break

print("\n--- 🎯 Examples of Relevant Documents (Qrels) ---")
count = 0
for qrel in dataset.qrels_iter():
    # qrel.doc_id هو رقم المستند الذي يعتبر إجابة صحيحة للـ query_id
    print(f"Query ID: {qrel.query_id} ---> Relevant Doc ID: {qrel.doc_id}")
    count += 1
    if count >= 5: # Printing only 5 examples
        break