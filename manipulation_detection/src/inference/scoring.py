import numpy as np

def calculate_risk_score(predictions):
    """
    Calculate overall risk score based on manipulation tactic probabilities.
    
    Args:
        predictions (dict): {label: probability}
        
    Returns:
        float: Risk score between 0.0 and 1.0
        str: Risk level (Low, Medium, High, Critical)
        str: Primary pattern detected
    """
    if not predictions:
        return 0.0, "Low", "None"

    # Define weights for different tactics (severity)
    severity_weights = {
        "threatening_intimidation": 1.0,
        "gaslighting": 0.9,
        "coercive_control": 0.95, # If present
        "belittling_ridicule": 0.8,
        "stonewalling": 0.7,
        "guilt_tripping": 0.6,
        "love_bombing": 0.6, # Can be high risk in context
        "passive_aggression": 0.5,
        "deflection": 0.5,
        "whataboutism": 0.4,
        "appeal_to_emotion": 0.4,
        "ethical_persuasion": 0.0,
        "neutral_conversation": 0.0,
        "coercive_control": 0.9
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
    
    # Determine level
    if risk_score > 0.8:
        level = "Critical"
    elif risk_score > 0.6:
        level = "High"
    elif risk_score > 0.3:
        level = "Medium"
    else:
        level = "Low"
        
    return risk_score, level, primary_pattern

def calculate_darvo_score(predictions):
    """
    Calculate DARVO score based on specific tactics.
    DARVO: Deny, Attack, Reverse Victim & Offender.
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
