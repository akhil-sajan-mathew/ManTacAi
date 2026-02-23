from .heuristics import apply_heuristics

def calculate_risk_score(predictions, context_factors=None, text_content="", semantic_data=None):
    """
    Calculate overall risk score based on manipulation tactic probabilities.
    Includes Heuristic Overrides (Phase 9/10) and Semantic Similarity (Phase 23).
    
    Args:
        predictions (dict): {label: probability}
        context_factors (list): Optional list of context strings
        text_content (str): Raw text for heuristic analysis
        semantic_data (tuple): (score, concept) from SemanticEngine
        
    Returns:
        float: Risk score between 0.0 and 1.0
        str: Risk level (Low, Medium, High, Critical)
        str: Primary pattern detected
        dict: Refined tactic scores
    """
    if not predictions:
        return 0.0, "Low", "None", {}

    # --- PHASE 9/10: HEURISTIC OVERRIDE ---
    # Apply rules to modify predictions based on keyword strength
    modified_probs, heuristic_pattern, heuristic_risk = apply_heuristics(text_content, predictions, context_factors, semantic_data=semantic_data)
    
    # Use modified probabilities for the rest of the calculation
    predictions = modified_probs

    # Define weights for different tactics (severity)
    severity_weights = {
        "threatening_intimidation": 1.0,
        "gaslighting": 0.9,
        "coercive_control": 0.95, # If present
        "belittling_ridicule": 0.8,
        "stonewalling": 0.7,
        "guilt_tripping": 0.6,
        "love_bombing": 0.4, # Reduced from 0.6 to avoid false positives on compliments
        "passive_aggression": 0.4, # Reduced from 0.5 to reduce FPs on ambiguous phrases
        "deflection": 0.5,
        "whataboutism": 0.4,
        "appeal_to_emotion": 0.4,
        "ethical_persuasion": 0.0,
        "neutral_conversation": 0.0,

        # V6 Normalcy Classes (Zero Risk)
        "benign_venting": 0.0,
        "healthy_conflict": 0.0,
        "benign_affection": 0.0,
        "neutral_logistics": 0.0,
        "urgent_emergency": 0.0
    }

    max_prob = 0.0
    primary_pattern = "None"
    weighted_risk_sum = 0.0
    
    for label, prob in predictions.items():
        weight = severity_weights.get(label, 0.5)
        risk_contribution = prob * weight
        
        if prob > max_prob:
            max_prob = prob
            primary_pattern = label
            
        weighted_risk_sum += risk_contribution

    # Normalize or cap risk score
    # Simple approach: Max probability weighted by severity
    risk_score = max_prob * severity_weights.get(primary_pattern, 0.5)
    
    # --- CONTEXT MODIFIERS (Phase 7) ---
    if context_factors:
        current_risk_float = float(risk_score)
        if "history_of_violence" in context_factors:
            # 30% Boost OR Min High Risk (0.7)
            current_risk_float = max(current_risk_float * 1.3, 0.7)
        if "financial_dependency" in context_factors:
            # 15% Boost OR Min Medium Risk (0.4)
            current_risk_float = max(current_risk_float * 1.15, 0.4)
        if "isolation" in context_factors:
            current_risk_float *= 1.2
        if "stalking_history" in context_factors:
            current_risk_float *= 1.25
        
        risk_score = min(current_risk_float, 1.0)
    
    # --- PHASE 23: SEMANTIC SIMILARITY OVERRIDE ---
    if semantic_data:
        sem_score, sem_concept = semantic_data
        # Contrastive score: how much MORE similar to danger than to safe
        # A positive score > threshold means the input is closer to danger concepts
        if sem_score > 0.50:  # Contrastive threshold (danger_sim - safe_sim)
            risk_score = max(risk_score, 0.95)
            primary_pattern = f"SEMANTIC_{sem_concept}"
    
    # Determine level
    if risk_score > 0.8:
        level = "Critical"
    elif risk_score > 0.6:
        level = "High"
    elif risk_score > 0.3:
        level = "Medium"
    else:
        level = "Low"
        
    return risk_score, level, primary_pattern, predictions

def calculate_darvo_score(predictions, text_blob=""):
    """
    Calculate DARVO score based on specific tactics.
    DARVO: Deny, Attack, Reverse Victim & Offender.
    Args:
        predictions (dict): Probability scores
        text_blob (str): Full text for keyword analysis (Optional)
    """
    if not predictions:
        return 0.0

    # 1. Deny (Gaslighting is key here)
    deny_score = (
        predictions.get("gaslighting", 0.0) + 
        predictions.get("deflection", 0.0) + 
        predictions.get("stonewalling", 0.0)
    )

    # 2. Attack
    attack_score = (
        predictions.get("belittling_ridicule", 0.0) + 
        predictions.get("threatening_intimidation", 0.0) + 
        predictions.get("passive_aggression", 0.0)
    )

    # 3. Reverse Victim & Offender
    reverse_score = (
        predictions.get("guilt_tripping", 0.0) + 
        predictions.get("appeal_to_emotion", 0.0) + 
        predictions.get("whataboutism", 0.0)
    )
    
    # --- KEYWORD BOOSTER (Calibration Fix) ---
    # Regex-like check for explicit reversal language
    # "Soft Multiplier" strategy: If existing reverse score > 0, amplify it.
    if text_blob:
        text_lower = text_blob.lower()
        reversal_triggers = ["i am the victim", "you are abusive", "actually you", "not me", "my fault"]
        
        # Check if ANY trigger exists
        if any(trigger in text_lower for trigger in reversal_triggers):
             # Boost Reverse score by 25% (Multiplier 1.25)
             # This prevents false positives on safe text (0 * 1.25 = 0)
             reverse_score *= 1.25

    # Cap components at 1.0
    deny_score = min(deny_score, 1.0)
    attack_score = min(attack_score, 1.0)
    reverse_score = min(reverse_score, 1.0)
    
    # Check for Synergy (The core of DARVO)
    # Count how many components are arguably present (> 0.15 threshold)
    components_present = 0
    if deny_score > 0.15: components_present += 1
    if attack_score > 0.15: components_present += 1
    if reverse_score > 0.15: components_present += 1
    
    # Base calculation (Average)
    raw_score = (deny_score + attack_score + reverse_score) / 3.0
    
    # Synergy Multipliers
    if components_present == 3:
        final_score = raw_score * 1.5 # Full DARVO = 50% boost
    elif components_present == 2:
        final_score = raw_score * 1.2 # Partial DARVO = 20% boost
    else:
        final_score = raw_score # Isolated tactic != DARVO
        
    return min(final_score, 1.0)
