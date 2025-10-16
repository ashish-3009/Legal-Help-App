import os
import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
from fpdf import FPDF
from sentence_transformers import SentenceTransformer, util
import torch
import re
from io import BytesIO  # For in-memory PDF

# --- Page setup ---
st.set_page_config(
    page_title="Legal Case Explorer",
    page_icon="⚖️",
    layout="wide"
)

# --- Custom CSS styling ---
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 38px !important;
            color: #2C3E50;
            font-weight: 700;
        }
        .case-card {
            background-color: #f9f9f9;
            padding: 18px 25px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        }
        mark {
            background-color: #FFF176;
            padding: 0 3px;
            border-radius: 3px;
        }
        .stButton>button {
            border-radius: 6px;
            background-color: #2E86C1;
            color: white;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #1B4F72;
            color: #f1f1f1;
        }
    </style>
""", unsafe_allow_html=True)

# --- Neo4j connection ---

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(user, password))

# --- Load embedding model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- Fetch cases from Neo4j ---
@st.cache_data
def fetch_cases():
    query = """
    MATCH (c:Case)
    RETURN c.id AS id,
           c.name AS name,
           c.summary AS summary,
           c.category AS category,
           c.court AS court,
           c.victim AS victim,
           c.defendant AS defendant,
           c.year AS year,
           c.verdict AS verdict,
           c.judge AS judge,
           c.lawyer AS lawyer
    """
    with driver.session() as session:
        results = session.run(query)
        data = [record.data() for record in results]
    return data

cases = fetch_cases()
case_df = pd.DataFrame(cases)

# --- Compute embeddings ---
@st.cache_data
def compute_case_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

case_texts = case_df['name'].astype(str) + ". " + case_df['summary'].astype(str)
case_embeddings = compute_case_embeddings(case_texts.tolist())

# --- Helper functions ---
def highlight_text(text, query):
    """Highlight matched words in text."""
    if not query:
        return text
    words = [re.escape(word) for word in query.split()]
    pattern = r"(" + "|".join(words) + r")"
    return re.sub(pattern, r"<mark>\1</mark>", text, flags=re.IGNORECASE)

def search_cases(query, filters, top_k=10):
    """Filter + semantic search."""
    filtered_df = case_df.copy()

    # Apply sidebar filters
    if filters["year"]:
        filtered_df = filtered_df[filtered_df["year"].astype(str) == filters["year"]]
    if filters["category"]:
        filtered_df = filtered_df[filtered_df["category"] == filters["category"]]
    if filters["court"]:
        filtered_df = filtered_df[filtered_df["court"] == filters["court"]]
    if filters["judge"]:
        filtered_df = filtered_df[filtered_df["judge"] == filters["judge"]]

    if not query:
        return filtered_df

    # Semantic search on filtered cases
    query_embedding = model.encode(query, convert_to_tensor=True)
    indices = filtered_df.index.tolist()
    subset_embeddings = case_embeddings[indices]
    cos_scores = util.cos_sim(query_embedding, subset_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))
    top_indices = [indices[i] for i in top_results.indices.cpu().numpy()]
    return filtered_df.loc[top_indices]

# --- Sidebar filters ---
st.sidebar.header("⚙️ Filters")
years = sorted(case_df['year'].dropna().astype(str).unique())
categories = sorted(case_df['category'].dropna().unique())
courts = sorted(case_df['court'].dropna().unique())
judges = sorted(case_df['judge'].dropna().unique())

filters = {
    "year": st.sidebar.selectbox("📅 Year", [""] + years),
    "category": st.sidebar.selectbox("📂 Category", [""] + categories),
    "court": st.sidebar.selectbox("🏛️ Court", [""] + courts),
    "judge": st.sidebar.selectbox("⚖️ Judge", [""] + judges)
}

# --- Main UI ---
st.markdown("<h1 class='title'>Legal Case Explorer ⚖️</h1>", unsafe_allow_html=True)
st.write("Search and explore legal cases using semantic understanding and advanced filters.")

search_query = st.text_input("🔍 Enter keywords or a question:")

filtered_cases = search_cases(search_query, filters)

st.markdown("---")
st.subheader("📄 Matching Cases")

# --- Display search results with inline expanders ---
if filtered_cases.empty:
    st.warning("No cases found. Try changing your filters or search query.")
else:
    for _, case in filtered_cases.iterrows():
        with st.container():
            st.markdown(f"""
            <div class='case-card'>
                <h4>{case['name']}</h4>
                <p><b>Category:</b> {case['category']} | <b>Year:</b> {case['year']} | <b>Court:</b> {case['court']}</p>
                <p>{highlight_text(case['summary'][:350] + '...', search_query)}</p>
            </div>
            """, unsafe_allow_html=True)

            # Inline expandable details
            with st.expander(f"📘 View Details: {case['name']}"):
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Category:** {case['category']}")
                    st.markdown(f"**Year:** {case['year']}")
                    st.markdown(f"**Court:** {case['court']}")
                    st.markdown(f"**Judge:** {case['judge']}")
                with cols[1]:
                    st.markdown(f"**Lawyer:** {case['lawyer']}")
                    st.markdown(f"**Victim:** {case['victim']}")
                    st.markdown(f"**Defendant:** {case['defendant']}")
                    st.markdown(f"**Verdict:** {case['verdict']}")

                st.markdown(
                    f"**Summary:** {highlight_text(case['summary'], search_query)}",
                    unsafe_allow_html=True
                )

                # --- PDF Export in-memory (fixed) ---
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for key, value in case.items():
                    pdf.multi_cell(0, 10, f"{key.capitalize()}: {value}")

                # Get PDF as bytes
                pdf_bytes = pdf.output(dest='S').encode('latin1')
                pdf_buffer = BytesIO(pdf_bytes)

                st.download_button(
                    label="📥 Download Case PDF",
                    data=pdf_buffer,
                    file_name=f"{case['name']}.pdf",
                    mime="application/pdf"
                )
