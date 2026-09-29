from retrieval_module import retrieve_relevant_docs
from generation_module import generate_verdict

def main():
    print("Interactive RAG CLI for Fake News Detection")
    print("Enter a news statement to check (or 'quit' to exit).\n")
    
    while True:
        query = input("Enter news: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a valid statement.\n")
            continue
        
        print("Processing...")
        
        # Retrieve top 3 facts + Wikipedia
        docs, scores = retrieve_relevant_docs(query, k=3)
        if not docs:
            print("No relevant facts found. Defaulting to Fake.\n")
            print("Verdict: **Fake** (No matches)")
            print("Explanation: This claim has no supporting evidence in verified sources.\n")
            continue
        
        # Normalize score to 0-10, boost if Wikipedia present (last score=5.0)
        avg_score = sum(scores) / len(scores)
        normalized_score = min(10.0, max(0.0, avg_score / 1.5))
        wikipedia_present = len(scores) > 3 and scores[-1] == 5.0
        if wikipedia_present:
            normalized_score += 3.5  # Aggressive boost for dynamic fact
            normalized_score = min(10.0, normalized_score)
        
        # Ultra-simple RAG prompt (minimal to avoid drift/echo)
        facts_str = "\n".join([f"- {doc[:80]}..." for doc in docs])
        prompt = f"""Claim: {query}

Relevant Real Facts:
{facts_str}

Verdict: """
        
        # Generate with score for balance
        verdict, explanation = generate_verdict(prompt, score=normalized_score)
        
        # Override: For high score + Wikipedia, force Real (trusts facts, bypasses generation issues)
        if normalized_score >= 6.0 and wikipedia_present:
            verdict = 'Real'
            explanation = f"Supported by high-relevance facts including Wikipedia (score: {normalized_score:.2f}). Matches the claim based on verified sources."
        # Echo detection only for low scores (force Fake if echo)
        elif normalized_score < 6.0:
            raw_lower = explanation.lower()
            query_words = query.lower().split()[:3]  # First 3 words of query
            echo_detected = any(word in raw_lower for word in query_words)
            if echo_detected and verdict == 'Real':
                verdict = 'Fake'
                explanation = f"Lacks evidence (echo detected, score: {normalized_score:.2f}). {explanation.split('.')[0]}..."
        # Downgrade for fakes with 'fake' in raw and low score
        elif verdict == 'Real' and 'fake' in explanation.lower() and normalized_score < 4.0:
            verdict = 'Fake'
            explanation = f"Lacks evidence (score: {normalized_score:.2f}). {explanation.split('.')[0]}..."
        
        # Display
        print(f"Verdict: **{verdict}** (Relevance Score: {normalized_score:.2f}/10)")
        print(f"Explanation: {explanation}\n")
        
        # Show facts
        print("Retrieved Real Facts (for reference):")
        for i, (doc, score) in enumerate(zip(docs, scores), 1):
            prefix = " (Wikipedia)" if i == len(docs) and score == 5.0 else ""
            print(f"{i}. {doc[:80]}... (BM25 Score: {score:.2f}){prefix}")
        print("=" * 50 + "\n")

if __name__ == "__main__":
    main()