# ============================================================
# Legal Help Knowledge Graph - Semantic Search (Normalized)
# ============================================================

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 1. Neo4j Connection Setup
# ---------------------------
uri = "neo4j+s://30d3f41f.databases.neo4j.io"
username = "neo4j"
password = "CQizjjgwq3AAHS0pYMoVL_QY1_a3CAWs2gitNlQ0VFM"  # update if different
driver = GraphDatabase.driver(uri, auth=(username, password))

# ---------------------------
# 2. Load SentenceTransformer Model
# ---------------------------
print("🔹 Loading SentenceTransformer model (MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully!\n")

# ---------------------------
# 3. Fetch Cases + Embeddings
# ---------------------------
def fetch_cases(tx):
    query = """
    MATCH (c:Case)
    WHERE c.embedding IS NOT NULL
    RETURN c.id AS id, c.name AS name, c.summary AS summary, c.category AS category, c.embedding AS embedding
    """
    result = tx.run(query)
    records = [r.data() for r in result]
    for r in records:
        # Convert list to numpy array and normalize
        emb = np.array(r["embedding"])
        r["embedding"] = emb / np.linalg.norm(emb)
    return records

with driver.session() as session:
    print("🔹 Fetching cases from Neo4j...")
    cases = session.execute_read(fetch_cases)
print(f"✅ Retrieved {len(cases)} cases with embeddings.\n")

# ---------------------------
# 4. Semantic Search Function
# ---------------------------
SIM_THRESHOLD = 0.4  # minimum cosine similarity to show

def search_cases(query, top_k=5):
    # 1️⃣ Embed and normalize the query
    query_emb = model.encode([query], convert_to_numpy=True)[0]
    query_emb = query_emb / np.linalg.norm(query_emb)

    # 2️⃣ Compute cosine similarity with all case embeddings
    similarities = []
    for case in cases:
        sim = float(np.dot(query_emb, case["embedding"]))  # cosine similarity
        if sim >= SIM_THRESHOLD:
            similarities.append((sim, case))

    # 3️⃣ Sort by highest similarity
    similarities.sort(key=lambda x: x[0], reverse=True)

    # 4️⃣ Return top-k matches
    return similarities[:top_k]

# ---------------------------
# 5. User Interaction Loop
# ---------------------------
print("🎯 Semantic Search Ready! Type your query below (type 'exit' to quit):\n")

while True:
    user_query = input("Enter your search query: ").strip()
    if user_query.lower() == "exit":
        print("👋 Exiting semantic search. Goodbye!")
        break

    results = search_cases(user_query, top_k=5)
    if not results:
        print(f"\n❌ No relevant cases found for: '{user_query}'\n")
        continue

    print(f"\n🔹 Top {len(results)} results for: '{user_query}'\n")
    for rank, (score, case) in enumerate(results, start=1):
        print(f"{rank}. {case['name']} [{case['category']}] - Score: {score:.3f}")
        print(f"   Summary: {case['summary'][:200]}{'...' if len(case['summary'])>200 else ''}\n")
