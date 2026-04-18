# =====================================================================
# 1. SETUP LLM API (The NLP Brain)
# =====================================================================
API_KEY = "ENTER_API_KEY_HERE" 
genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel('gemini-2.5-flash')

def parse_with_real_ai(user_sentence, valid_categories):
    prompt = f"""
    The user said: "{user_sentence}"
    
    You are a classification engine mapping text to a STRICT predefined vector space. 
    You MUST ONLY use categories from this exact list:
    {valid_categories}
    
    Analyze the user's intent:
    - If they LIKE or WANT something, give it a POSITIVE weight (0.1 to 1.0).
    - If they DISLIKE or NEGATE something, give it a NEGATIVE weight (-0.1 to -1.0).
    
    Return ONLY a valid JSON dictionary with the top 2 to 4 impacted categories and their float weights. 
    DO NOT output any markdown, explanations, or text outside the JSON object.
    """
    try:
        response = llm_model.generate_content(prompt)
        clean_text = response.text.strip().strip('json').strip('').strip()
        impact_weights = json.loads(clean_text)
        strict_weights = {k: v for k, v in impact_weights.items() if k in valid_categories}
        return strict_weights
    except Exception as e:
        print(f"\n[System Notice: The NLP Engine couldn't parse that sentence cleanly.]")
        return {}
