# ============================================================
# Legal Help Knowledge Graph - Embedding Script (Final Working)
# Creates embeddings for all cases and categories in Neo4j
# ============================================================

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import time

# ---------------------------
# 1. Neo4j Connection Setup
# ---------------------------
uri = "bolt://127.0.0.1:7687"  # use bolt:// for local Neo4j
username = "neo4j"
password = "CQizjjgwq3AAHS0pYMoVL_QY1_a3CAWs2gitNlQ0VFM"         # update if different
driver = GraphDatabase.driver(uri, auth=(username, password))

# ---------------------------
# 2. Load SentenceTransformer Model
# ---------------------------
print("🔹 Loading SentenceTransformer model (MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully!\n")

# ---------------------------
# 3. Fetch All Cases from Neo4j
# ---------------------------
def fetch_case_data(tx):
    query = """
    MATCH (c:Case)
    RETURN c.id AS id, c.name AS name, c.summary AS summary, c.category AS category
    """
    result = tx.run(query)
    return pd.DataFrame([r.data() for r in result])

with driver.session() as session:
    print("🔹 Fetching case data from Neo4j...")
    df = session.execute_read(fetch_case_data)

print(f"✅ Retrieved {len(df)} cases from Neo4j\n")

# ---------------------------
# 4. Generate Embeddings in Batches
# ---------------------------
def generate_embeddings(texts, batch_size=64):
    """Generate embeddings for all texts in batches"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True)
        all_embeddings.extend(emb)
        print(f"✅ Embedded {i + len(batch)} / {len(texts)} cases...")
    return np.array(all_embeddings)

print("🔹 Generating embeddings for all summaries...")
start_time = time.time()
embeddings = generate_embeddings(df["summary"].fillna("").tolist())
df["embedding"] = embeddings.tolist()  # ✅ FIXED: proper list conversion
print(f"✅ Completed embeddings in {round(time.time() - start_time, 2)} seconds\n")

# ---------------------------
# 5. Store Case Embeddings Back to Neo4j
# ---------------------------
def store_embeddings(tx, data_batch):
    """Write embeddings for a batch of cases"""
    for record in data_batch:
        tx.run("""
            MATCH (c:Case {id: $id})
            SET c.embedding = $embedding
        """, id=record["id"], embedding=record["embedding"])  # ✅ FIXED: removed .tolist()

print("🔹 Writing embeddings to Neo4j...")
with driver.session() as session:
    BATCH_SIZE = 100
    for i in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        batch_data = batch_df[["id", "embedding"]].to_dict("records")
        session.execute_write(store_embeddings, batch_data)
        print(f"✅ Stored embeddings for {i + len(batch_df)} / {len(df)} cases...")  # ✅ FIXED line

print("✅ All case embeddings stored successfully in Neo4j!\n")

# ---------------------------
# 6. Compute and Store Category-Level Embeddings
# ---------------------------
def store_category_embedding(tx, category, embedding):
    """Store average embedding per category"""
    tx.run("""
        MERGE (cat:Category {name: $category})
        SET cat.embedding = $embedding
    """, category=category, embedding=embedding)

print("🔹 Computing average embeddings per category...")
with driver.session() as session:
    categories = df["category"].unique()
    for cat in categories:
        subset = df[df["category"] == cat]["embedding"].tolist()
        if subset:
            avg_emb = np.mean(np.array(subset), axis=0)
            session.execute_write(store_category_embedding, cat, avg_emb.tolist())
            print(f"✅ Category embedding stored for: {cat}")

print("\n🎉 All embeddings (case + category) added successfully to Neo4j!")
driver.close()
