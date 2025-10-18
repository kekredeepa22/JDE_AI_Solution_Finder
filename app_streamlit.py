import openai
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os

# --- 1. Configuration ---
load_dotenv()

CSV_FILE = "JDE_Issues.csv"
ISSUE_COLUMN = "Issue"
SOLUTION_COLUMN = "Resolution"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
openai.api_key = st.secrets["general"]["openai_api_key"]
client = OpenAI(api_key=openai.api_key)

# --- 2. Streamlit UI ---
st.set_page_config(page_title="JDE AI Solution Finder", layout="wide")
st.title("🧠 JDE AI Solution Finder")
st.caption("Find resolutions to JDE technical issues using a knowledge base powered by OpenAI embeddings.")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(CSV_FILE, encoding="latin-1")
    df = df[[ISSUE_COLUMN, SOLUTION_COLUMN]].dropna()
    df[ISSUE_COLUMN] = df[ISSUE_COLUMN].astype(str).str.strip()
    df[SOLUTION_COLUMN] = df[SOLUTION_COLUMN].astype(str).str.strip()
    df = df[df[ISSUE_COLUMN] != ""]
    return df

df = load_data()
st.success(f"✅ Loaded {len(df)} issues from the knowledge base.")

# --- 3. Create and Cache Embeddings ---
@st.cache_resource(show_spinner=True)
def create_embeddings(texts):
    embeddings = []
    for i in range(0, len(texts), 50):  # batch processing for efficiency
        batch = texts[i:i+50]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([e.embedding for e in response.data])
    return np.array(embeddings)

with st.spinner("Creating embeddings for the knowledge base (first-time setup)..."):
    df["embedding"] = list(create_embeddings(df[ISSUE_COLUMN].tolist()))
st.success("✅ Knowledge base embeddings ready!")

# --- 4. Helper: Find Top Matches ---
def find_similar_issues(query, top_k=3):
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    similarities = cosine_similarity([q_emb], list(df["embedding"]))[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return df.iloc[top_indices]

# --- 5. Core LLM Query ---
def get_resolution_from_openai(user_query: str):
    similar_issues = find_similar_issues(user_query, top_k=3)
    
    if similar_issues.empty:
        return "No relevant resolution found in the database. Please refer to official JDE documentation."

    # Build the knowledge base context
    context = "\n".join(
        f"Issue: {row[ISSUE_COLUMN]}\nResolution: {row[SOLUTION_COLUMN]}"
        for _, row in similar_issues.iterrows()
    )

    system_prompt = (
        "You are a JDE technical support assistant. "
        "You are provided with a knowledge base that contains real issues and their verified resolutions. "
        "Your ONLY job is to find and present the correct resolution from this knowledge base. "
        "Do NOT hallucinate, use any external knowledge or assumptions. "
        "If the user's issue does not closely match any issue in the knowledge base, "
        "reply exactly with: 'No relevant resolution found in the database. Please refer to official JDE documentation.' "
        "Do not attempt to generate, guess, or infer a solution that is not explicitly mentioned in the provided data. "
        "If a match exists, provide the corresponding resolution in a clear and concise manner."
    )

    user_message = f"USER ISSUE:\n{user_query}\n\nRELEVANT CONTEXT:\n{context}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=400
    )

    return response.choices[0].message.content.strip(), similar_issues

# --- 6. Streamlit Interaction ---
st.divider()
st.subheader("🔍 Search for a JDE Issue")

query = st.text_area("Enter your JDE technical issue:", height=100, placeholder="e.g., My UBE report is not printing in JDE...")

if st.button("Find Resolution"):
    if not query.strip():
        st.warning("⚠️ Please enter an issue description.")
    else:
        with st.spinner("Searching for similar issues and generating response..."):
            result, similar_issues = get_resolution_from_openai(query)
        
        st.subheader("💡 AI-Generated Resolution")
        st.write(result)

        st.subheader("📋 Top Matching Issues from Knowledge Base")
        for i, row in similar_issues.iterrows():
            with st.expander(f"Issue: {row[ISSUE_COLUMN]}"):
                st.write(f"**Resolution:** {row[SOLUTION_COLUMN]}")

st.divider()
st.caption("Built with ❤️ using OpenAI and Streamlit.")


