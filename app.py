import streamlit as st
from retrieval_module import retrieve_relevant_docs
from generation_module import generate_verdict
import os
import numpy as np  # For calculations
import traceback  # For error details

# Page config (title, favicon, layout)
st.set_page_config(
    page_title="FactCheck AI - Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (mimics your HTML gradient and styles)
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #F0F4F8, #93C5FD, #FBCFE8);
        background-size: 400% 400%;
        animation: gradient-animation 15s ease infinite;
    }
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stButton > button {
        background: linear-gradient(135deg, #8B5CF6, #A78BFA);
        color: white;
        border-radius: 9999px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }
    .metric-container {
        background: rgba(255,255,255,0.8);
        padding: 1rem;
        border-radius: 1rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description (like your HTML h1)
st.title("📰 Fake News Detection System")
st.markdown("**Powered by AI & Retrieval-Augmented Generation (RAG)**")
st.markdown("Enter a news headline below to get a fact-check verdict using real facts from a knowledge base.")

# Sidebar (project info, like your footer)
with st.sidebar:
    st.header("📋 Project Info")
    st.write("""
    - **Data**: Politifact Fake/Real (~1000 samples, preprocessed).
    - **Retrieval**: BM25 on real news corpus (~400 samples) + Wikipedia dynamic facts.
    - **Generation**: GPT-2 (upgraded for better reasoning).
    - **Setup**: Run `python retrieval_module.py` first to build index.
    """)
    st.markdown("---")
    st.info("© 2025 FactCheck AI. Built with Streamlit & Transformers.")

# Session state to prevent rerun spam (store results per input)
if 'results' not in st.session_state:
    st.session_state.results = None

# Main Content
# Input Section (like your textarea)
news_input = st.text_area(
    "Enter News Headline:",
    placeholder="e.g., Obama secretly funds ISIS",
    height=100,
    help="Paste a news title or short article excerpt for analysis."
)

# Process Button (like your "Check News") with session state to avoid reruns
if st.button("🔍 Check News", type="primary"):
    if not news_input.strip():
        st.warning("Please enter a news headline first.")
    else:
        # Clear previous results
        st.session_state.results = None
        st.rerun()  # Trigger rerun to process

# Show results if input changed or button clicked (use session state to cache)
if news_input.strip() and st.session_state.results is None:
    with st.spinner("Retrieving real facts and generating verdict..."):
        try:
            # Step 1: Retrieve relevant real facts (unpack tuple: docs, scores)
            docs, scores = retrieve_relevant_docs(news_input, k=3)
            if not docs:
                st.session_state.results = {"verdict": "Fake", "explanation": "No relevant facts found. Defaulting to Fake.", "score": 0.0, "docs": [], "scores": [], "wikipedia_present": False}
            else:
                # Calculate average score from scores list
                avg_score = sum(scores) / len(scores) if scores else 0
                facts_text = "\n".join(docs)  # docs is list of strings
                
                # Bulletproof logic: Normalize score with Wikipedia boost (if last score ==5.0)
                normalized_score = min(10.0, max(0.0, avg_score / 1.5))
                wikipedia_present = len(scores) > 3 and scores[-1] == 5.0
                if wikipedia_present:
                    normalized_score += 3.5  # Aggressive boost for dynamic fact
                    normalized_score = min(10.0, normalized_score)
                
                # Step 2: Build prompt with retrieved facts
                prompt = f"""You are a fact-checker. Use the real facts to evaluate the news.

Real facts: {facts_text}

News: "{news_input}"

Verdict: """
                
                # Step 3: Generate verdict/explanation (pass normalized_score for fallback)
                verdict, explanation = generate_verdict(prompt, score=normalized_score)
                
                # Step 4: Bulletproof override (force Real for high score + Wikipedia)
                if normalized_score >= 6.0 and wikipedia_present:
                    verdict = 'Real'
                    explanation = f"Supported by high-relevance facts including Wikipedia (score: {normalized_score:.2f}). Matches the claim based on verified sources."
                # Fallback - Low relevance biases to Fake
                elif avg_score < 3.0:
                    verdict = "Fake"
                    explanation = explanation or "Low relevance in facts; likely unsubstantiated."
                # Echo detection for low scores (force Fake if echo)
                elif avg_score < 6.0:
                    raw_lower = explanation.lower()
                    query_words = news_input.lower().split()[:3]
                    echo_detected = any(word in raw_lower for word in query_words)
                    if echo_detected and verdict == 'Real':
                        verdict = 'Fake'
                        explanation = f"Lacks evidence (echo detected, score: {normalized_score:.2f}). {explanation.split('.')[0]}..."
                
                # Store in session state to prevent rerun spam
                st.session_state.results = {
                    "verdict": verdict,
                    "explanation": explanation,
                    "normalized_score": normalized_score,
                    "docs": docs,
                    "scores": scores,
                    "wikipedia_present": wikipedia_present
                }
                st.rerun()  # Rerun to show results
        except Exception as e:
            st.error(f"Error during analysis: {e}\n{traceback.format_exc()}")
            st.info("Ensure you've run `python retrieval_module.py` and `python generation_module.py` to load the model/retriever.")
            st.session_state.results = None

# Display results if available (from session state)
if st.session_state.results:
    results = st.session_state.results
    st.subheader("📊 Results")
    
    # Columns: Left (Verdict + Explanation), Right (Score + Facts)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Verdict Badge (color-coded, like your verified/error)
        if results["verdict"] == "Real":
            st.success(f"**Verdict: {results['verdict']}**")
            st.balloons()  # Fun animation
        else:
            st.error(f"**Verdict: {results['verdict']}**")
        
        st.markdown("**Explanation:**")
        st.write(results["explanation"])
    
    with col2:
        # Relevance Score (use normalized_score for UI metric)
        st.metric("Relevance Score", f"{results['normalized_score']:.2f}/10", help="Higher = Better match to real facts (BM25 + Wikipedia boost).")
        
        st.markdown("**Top Supporting Real Facts:**")
        for i, (doc, score) in enumerate(zip(results["docs"], results["scores"]), 1):
            prefix = " (Wikipedia)" if i == len(results["docs"]) and score == 5.0 else ""
            with st.expander(f"{i}. Score: {score:.2f}{prefix}"):
                st.write(doc)
    
    # Additional Actions (like your copy/share buttons)
    col3, col4 = st.columns(2)
    with col3:
        if st.button("📋 Copy Result"):
            st.code(f"Verdict: {results['verdict']}\nExplanation: {results['explanation']}\nScore: {results['normalized_score']:.2f}")
    with col4:
        if st.button("🔗 Share"):
            st.info("Share via: Copy the result above and paste in social/email.")
    
    # Model Explanation (like your HTML section)
    st.markdown("---")
    st.subheader("🤖 How It Works")
    st.write("""
    The system uses:
    - **Retrieval (BM25)**: Searches a database of verified real news for matching facts.
    - **Generation (GPT-2)**: Analyzes retrieved facts with GPT-2 to produce a verdict and explanation.
    - **Wikipedia Boost**: Adds dynamic facts for latest/global news (e.g., Indian events).
    - **Bulletproof Override**: Forces 'Real' for high-score matches with Wikipedia (trusts facts over generation).
    - **Fallback**: Low score defaults to 'Fake' for caution.
    """)
    
    # Clear button to reset for new query
    if st.button("🗑️ New Query"):
        st.session_state.results = None
        st.rerun()

# Footer (like your HTML footer)
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #718096;'>"
    "© 2024 FactCheck AI. All rights reserved. | "
    "<a href='#'>About</a> | "
    "<a href='#'>Privacy</a> | "
    "<a href='#'>Contact</a>"
    "</div>",
    unsafe_allow_html=True
)

# Run Instructions (if app loads)
if __name__ == "__main__":
    # Auto-run if needed, but Streamlit handles
    pass