import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


df = pd.read_csv("edamam_full_branched_clean_txt.csv").head(100000)
corpus = {}
for _, row in df.iterrows():
    doc_id = str(row["id"])
    corpus[doc_id] = {
        "title": row["title"],
        "text": row["text"]
    }


if not os.path.exists("doc_embeddings.pt"):
    print("❌ Error: doc_embeddings.pt not found. Please run the preprocessing script first.")
    exit()




saved = torch.load("doc_embeddings.pt")
doc_ids = saved["doc_ids"]
doc_embeddings = saved["doc_embeddings"]

# ✅ Load the same embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device="cpu")

# === 🔁 Start interactive query ===
print("\n💡 Enter command: query <keyword> | see <doc_id> | end")

while True:
    user_input = input("\n>>> ").strip()

    if user_input.lower() == "end":
        print("👋 Program terminated.")
        break

    elif user_input.lower().startswith("query "):
        query_text = user_input[6:].strip()
        if not query_text:
            print("⚠️ Please enter a valid query.")
            continue

        query_embedding = embedding_model.encode([query_text], convert_to_tensor=True)
        cos_sim = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings, dim=1)
        scores = cos_sim.cpu().numpy()
        top_k = 5
        top_indices = scores.argsort()[::-1][:top_k]

        print(f"\n🔍 Query results for \"{query_text}\":")
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = doc_ids[idx]
            score = scores[idx]
            print(f"{rank}. {doc_id} | {corpus[doc_id]['title']} | Score: {score:.4f}")

    elif user_input.lower().startswith("see "):
        doc_id = user_input[4:].strip()
        if not doc_id.endswith(".txt") and doc_id + ".txt" in corpus:
            doc_id += ".txt"  # Auto-complete .txt
        if doc_id in corpus:
            print(f"\n📄 Document {doc_id} content:")
            print(f"[Title] {corpus[doc_id]['title']}")
            print(f"[Text] {corpus[doc_id]['text']}")
        else:
            print(f"⚠️ Document ID '{doc_id}' not found.")

    else:
        print("⚠️ Invalid command. Use: query <keyword> / see <doc_id> / end")
