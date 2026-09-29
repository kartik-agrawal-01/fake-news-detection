import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import os
import pickle

# Set device to CPU
device = torch.device("cpu")
print(f"Device set to use {device}")

# Load the model and tokenizer (base for stability)
model_name = "gpt2"  # Base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.to(device)
model.eval()

print(f"Base model loaded from '{model_name}'!")

def generate_verdict(prompt, max_length=100, temperature=0.7, repetition_penalty=2.0, score=None):
    """
    Generate a verdict and explanation with anti-echo sampling.
    Returns (verdict, explanation).
    """
    # Add simple guidance if prompt lacks "Verdict: "
    if "Verdict:" not in prompt:
        guided_prompt = f"{prompt}\nVerdict: Fake or Real. Explanation: "
    else:
        guided_prompt = prompt
    
    inputs = tokenizer.encode(guided_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Bad words IDs to prevent echo and repetition (expanded for prompt structure)
    bad_words = ["Source:", "Fact:", "Relevant Real Facts:", "You are a fact-checker", "If facts match", "Use only these", "Verdict:", "Obama administration", "funding the Islamic State", "The White House", "The truth is", "Example", "Claim:"]
    bad_token_ids = []
    for word in bad_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if tokens:
            bad_token_ids.append(tokens)  # Full phrase as list of token_ids
    bad_words_ids = bad_token_ids if bad_token_ids else None  # List of lists of ints
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            min_new_tokens=20,  # Avoid empty
            temperature=temperature,  # Moderate for diversity
            repetition_penalty=repetition_penalty,  # High to reduce loops
            num_beams=2,  # Beam for structure
            do_sample=True,  # Sampling to break loops
            early_stopping=False,  # Full generation
            bad_words_ids=bad_words_ids,  # Block phrases
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = full_output[len(guided_prompt):].strip()
    
    print(f"[DEBUG] Raw generated text: '{generated_text}'")
    
    if not generated_text:
        print("[DEBUG] Empty generation - defaulting to Fake.")
        return 'Fake', "No supporting evidence found in facts; treated as unverified."
    
    # Aggressive clean repetition (dedup sentences and echo)
    generated_text = re.sub(r'(\bFact:\s*[^.]*?\.)\s*\1+', r'\1', generated_text, flags=re.IGNORECASE)
    generated_text = re.sub(r'(\bSource:\s*[^.]*?\.)\s*\1+', r'\1', generated_text, flags=re.IGNORECASE)
    generated_text = re.sub(r'(\bThe truth is\s*[^.]*?\.)\s*\1+', r'\1', generated_text, flags=re.IGNORECASE)
    #generated_text = re.sub(r'(\bThe White House\s*[^.]*?\.)\s*\1+', r'\1', generated_text, flags=re.IGNORECASE)
    generated_text = re.sub(r'_{2,}', ' ', generated_text)  # Remove underscores
    generated_text = re.sub(r'(http[^ \n]+)', '[Source Link]', generated_text)  # Neutralize links
    
    # Regex parsing
    verdict_match = re.search(r'(?i)(verdict:\s*)(fake|real|true|false)', generated_text)
    if verdict_match:
        verdict = verdict_match.group(2).capitalize()
        if verdict.lower() in ['true', 'real']:
            verdict = 'Real'
        elif verdict.lower() in ['false', 'fake']:
            verdict = 'Fake'
        explanation_start = verdict_match.end()
        explanation = generated_text[explanation_start:explanation_start+100].strip()
        if len(explanation) > 100:
            explanation = explanation[:100] + "..."
        if not explanation or len(explanation) < 10:
            explanation = "Analysis based on available facts."
        print(f"[DEBUG] Regex parsed: {verdict}, {explanation[:50]}...")
        return verdict, explanation
    
    # Enhanced keyword fallback (add 'echo' detection for hallucination)
    gen_lower = generated_text.lower()
    accused_count = gen_lower.count('accused')
    echo_count = gen_lower.count('youtube') + gen_lower.count('http') + gen_lower.count('source:')
    fake_signals = gen_lower.count('fake') + gen_lower.count('false') + gen_lower.count('no evidence') + gen_lower.count('conspiracy') + gen_lower.count('baseless') + gen_lower.count('untrue') + gen_lower.count('hoax') + gen_lower.count('not') + gen_lower.count('no') + gen_lower.count('contradicts') + gen_lower.count('failed') + accused_count + echo_count
    real_signals = gen_lower.count('real') + gen_lower.count('true') + gen_lower.count('confirmed') + gen_lower.count('matches') + gen_lower.count('supported') + gen_lower.count('verified') + gen_lower.count('is true') + gen_lower.count('signed') + gen_lower.count('passed') + gen_lower.count('law') + gen_lower.count('bill') + gen_lower.count('implemented') + gen_lower.count('president') + gen_lower.count('election') + gen_lower.count('attack') + gen_lower.count('war')
    
    # Score override for balance (if passed from CLI)
    if score is not None and score >= 5.0 and real_signals > fake_signals:
        verdict = 'Real'
        explanation = f"Supported by high-relevance facts (score: {score:.1f}, keywords: {real_signals}). {generated_text[:80]}..."
    elif real_signals > fake_signals:
        verdict = 'Real'
        explanation = f"Supported by facts (keywords: {real_signals}). {generated_text[:80]}..."
    elif fake_signals > 1:
        verdict = 'Fake'
        explanation = f"Lacks evidence (keywords: {fake_signals}). {generated_text[:80]}..."
    else:
        verdict = 'Fake'  # Default
        explanation = f"Lacks supporting facts (keywords: {fake_signals}). {generated_text[:80]}..."
    
    print(f"[DEBUG] Keyword fallback: Fake={fake_signals}, Real={real_signals} → {verdict}")
    return verdict, explanation

# Save config
config = {
    'model_name': model_name,
    'device': str(device),
    'max_new_tokens': 100,
    'temperature': 0.7,
    'repetition_penalty': 2.0,
    'do_sample': True,
    'bad_words_ids': True  # Anti-echo
}
with open('generator_config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("Generator config saved as 'generator_config.pkl'. Ready for use!")

print("Generation module ready! Import generate_verdict for use.")

# Temp standalone test (remove after)
# test_prompt = """You are a fact-checker.
# News: "Obama secretly funds ISIS"
# Facts: No evidence of funding.
# Verdict: """
# verdict, explanation = generate_verdict(test_prompt)
# print(f"\nStandalone Test: Verdict={verdict}, Explanation={explanation}")