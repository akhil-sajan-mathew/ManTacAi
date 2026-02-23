
import re
from utils.nlp_utils import nlp_processor

def apply_heuristics(text, ai_predictions, context_factors=None, semantic_data=None):
    """
    Applies rule-based overrides to AI predictions to catch specific
    high-risk patterns and enforce Negative Constraints.
    """
    # Normalize text (handle smart quotes, apostrophes)
    text_lower = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    
    # Phase 22: Generate Lemmas for Smart Matching
    text_lemma = nlp_processor.lemmatize_text(text_lower)
    
    # Defaults
    primary_override = None
    risk_override = None
    
    # Copy predictions to avoid mutation issues
    modified_preds = ai_predictions.copy()

    # --- 1. NEGATIVE CONSTRAINTS (Sick/Tired) ---
    # Phase 18 Fix: Anti-Dampening (Trojan Horse Defense)
    # Old: Force 0.05. New: Dampen by 90% (x0.1) ONLY if not Critical.
    pain_patterns = [
        r"my back(?:.*)hurt", r"my head(?:.*)hurt", r"i'?m tired", r"i'?m exhaust",
        r"i feel sick", r"i(?:.*)in pain", r"everything(?:.*)hurt"
    ]
    if any(re.search(p, text_lemma) for p in pain_patterns):
        current_lb = modified_preds.get("love_bombing", 0)
        # Safety Floor: If ML predicts CRITICAL Love Bombing (>0.85), don't suppress.
        # (Rare for Love Bombing, but good practice).
        if current_lb < 0.85:
            modified_preds["love_bombing"] = current_lb * 0.1 # Dampen by 90%
            
        # Boost Guilt Tripping (Martyrdom context)
        current_guilt = modified_preds.get("guilt_tripping", 0)
        modified_preds["guilt_tripping"] = max(current_guilt, 0.65)
        # Note: We allow Guilt to override because it's a Risk-increasing flag, not a Safety flag.
        primary_override = "guilt_tripping"


    # --- 2. COERCIVE CONTROL (Ultimatums) ---
    # DANGER SIGNAL -> Keep Hard Override
    ultimatum_patterns = [
        r"it'?s either him or me", r"choose.*him or me", r"him or me.*choose",
        r"block him right now", r"delete his number", r"i forbid you",
        r"you represent me", r"don'?t you dare"
    ]
    if any(re.search(p, text_lemma) for p in ultimatum_patterns):
        modified_preds["coercive_control"] = 0.95
        primary_override = "coercive_control"
        risk_override = 0.95

    # --- 3. MONITORING / STALKING ---
    # DANGER SIGNAL -> Keep Boost
    monitoring_patterns = [
        r"who was that guy", r"who were you talking to", r"show me your phone",
        r"send me a screenshot", r"share your location", r"where are you right now"
    ]
    if any(re.search(p, text_lemma) for p in monitoring_patterns):
        current = modified_preds.get("coercive_control", 0)
        modified_preds["coercive_control"] = max(current, 0.75)
        if modified_preds["coercive_control"] > 0.7:
             primary_override = "coercive_control"

    # --- 4. DENIAL / GASLIGHTING ---
    # Phase 18 Fix: Weighted Dampening
    denial_patterns = [
        r"^i'?m not", r"i am not", r"^i never", r"you'?re imagining",
        r"stop making things up", r"that never happened"
    ]
    if any(re.search(p, text_lemma) for p in denial_patterns):
        current_lb = modified_preds.get("love_bombing", 0)
        # Safety Floor: If it's EXTREMELY kind words but also denial? Unlikely, but safe to dampen.
        if current_lb < 0.85:
             modified_preds["love_bombing"] = current_lb * 0.2 # Dampen by 80%
            
        current_gas = modified_preds.get("gaslighting", 0)
        modified_preds["gaslighting"] = max(current_gas, 0.65)
        primary_override = "gaslighting"

    # --- 5. THREATS ---
    # Explicit Violence -> CRITICAL OVERRIDE
    threat_patterns = [
        r"i will kill", r"i be kill", r"break your", r"watch your back", r"regret it", r"destroy you",
        r"(?:i'm|i am|i be)\s+(?:going to|gonna|go to)\s+(?:find|get|hurt|kill|end)\s+you",
        r"end your life", r"take your life", r"wish you die", r"hope you die"
    ]
    if any(re.search(p, text_lemma) for p in threat_patterns):
        modified_preds["threatening_intimidation"] = 0.98
        primary_override = "threatening_intimidation"
        risk_override = 0.98

    # --- 5.5. IMPLICIT / CONDITIONAL THREATS (Phase 24) ---
    # Restriction of movement, implied doom — contextually threatening
    # Uses both contraction and lemmatized forms (SpaCy expands contractions)
    implicit_threat_patterns = [
        r"(?:not|never)\s+make it\s+to\s+your",   # "won't make it to your car"
        r"(?:not|never)\s+make it\s+out",          # "won't make it out"
        r"(?:not|never)\s+(?:let|allow)\s+you\s+(?:leave|go)",  # "I won't let you leave"
        r"stop you from\s+(?:leaving|going)",      # "I will stop you from leaving"
        r"(?:not|never)\s+leave\s+this",           # "you're not leaving this house"
        r"(?:will|be)\s+stop\s+you",               # lemmatized "I will stop you"
    ]
    if any(re.search(p, text_lemma) for p in implicit_threat_patterns):
        modified_preds["coercive_control"] = max(modified_preds.get("coercive_control", 0), 0.95)
        primary_override = "coercive_control"
        risk_override = 0.95

    # --- 6. MARTYRDOM / NEGATIVE SERVICE ---
    # Phase 18 Fix: Anti-Dampening
    martyr_override = check_martyrdom_complex(text_lemma)
    if martyr_override:
        if martyr_override.get("suppress_tactic"):
            suppress = martyr_override["suppress_tactic"].lower().replace(" ", "_")
            current_suppress = modified_preds.get(suppress, 0)
            if current_suppress < 0.85: # Safety Floor
                modified_preds[suppress] = current_suppress * 0.1 
        
        override_label = martyr_override["override_tactic"].lower().replace(" ", "_")
        current_val = modified_preds.get(override_label, 0)
        boost_val = martyr_override.get("risk_modifier", 0.65)
        modified_preds[override_label] = max(current_val, boost_val)
        primary_override = override_label

    # --- 7. FINANCIAL DEFENSE (Phase 18 Fix) ---
    fin_def = check_financial_defense(text_lemma)
    if fin_def:
        cc_score = modified_preds.get("coercive_control", 0)
        if cc_score > 0.85:
            pass # Safety Floor
        else:
            suppress = fin_def["suppress_tactic"].lower().replace(" ", "_")
            if suppress in modified_preds:
                 modified_preds[suppress] = modified_preds[suppress] * 0.2

    # --- 8. ACCOUNTABILITY DEFENSE (Phase 19 Future-Proofing) ---
    # "We agreed" / "You promised"
    accountability = check_accountability_defense(
        text_lemma, 
        risk_override or max(modified_preds.values()),
        semantic_data=semantic_data
    )
    if accountability:
        if accountability.get("override") is False:
             # VIOLENCE/HIGH RISK DETECTED - ABORT DAMPENING
             pass 
        else:
             # Apply Dampening
             mod_factor = accountability.get("risk_modifier", 0.5)
             override_label = accountability.get("override_tactic")
             
             # Dampen all negative tactics
             for k in modified_preds:
                 modified_preds[k] *= mod_factor
             
             # Set primary label to Accountability Check (Safe)
             primary_override = override_label
             # Note: We don't override Primary Label if we are just suppressing.
            # But we effectively lower the risk.
             # Helper can return specific risk score if needed, or we just rely on dampening

    return modified_preds, primary_override, risk_override

def check_martyrdom_complex(text_lower):
    """
    Detects 'Act of Service' + 'Guilt Inverter' combination.
    """
    # 1. The "Act of Service" (The bait)
    service_keywords = [
        r"do it myself", r"handle it", r"clean it", r"fix it", 
        r"take care of it", r"pick it up", r"do everything"
    ]
    
    # 2. The "Guilt Inverter" (The hook)
    guilt_markers = [
        r"since you won'?t", r"since you be too", r"since you'?re so", 
        r"guess i have to", r"no one else will", r"don'?t worry about me", 
        r"used to it", r"fine", r"since no one help"
    ]

    has_service = any(re.search(k, text_lower) for k in service_keywords)
    has_guilt = any(re.search(k, text_lower) for k in guilt_markers)

    if has_service and has_guilt:
        return {
            "override_tactic": "GUILT TRIPPING",
            "suppress_tactic": "LOVE BOMBING",
            "risk_modifier": 0.65 
        }
    
    return None

def check_financial_defense(text_lower):
    """
    Prevents 'Coercive Control' flags when a victim is explaining 
    financial boundaries or lack of funds.
    """
    defense_triggers = [
        r"wait for my paycheck", r"can'?t afford", r"not in the budget",
        r"don'?t have the money", r"until i get pay", r"payday"
    ]
    
    if any(re.search(t, text_lower) for t in defense_triggers):
        return {
            "override_tactic": "Financial Boundary",
            # REMOVED HARD OVERRIDE (0.1)
            # Logic is now handled by caller (apply_heuristics) via Dampening
            "suppress_tactic": "Coercive Control"
        }
    return None

def check_accountability_defense(text_lower, original_ml_score, semantic_data=None):
    """
    Phase 19: Future-Proof Anti-Trojan Logic.
    Dampens risk for "Accountability" phrases ONLY if safe.
    """
    triggers = [
        r"we agreed", r"you promised", r"you said you would", r"our plan",
        r"broke that promise", r"handle the"
    ]
    
    # 1. CHECK TRIGGERS
    if not any(re.search(t, text_lower) for t in triggers):
        return None

    # 2. VIOLENCE BLACKLIST (The Hard Stop)
    violence_keywords = [
        r"kill", r"die", r"hurt", r"punch", r"weapon", r"shut up", 
        r"liar", r"end your life", r"destroy", r"death",
        # Phase 24: Implicit threat phrases (restriction of movement / implied doom)
        # Include both contraction and lemmatized forms (SpaCy expands contractions)
        r"won't make it", r"will not make it", r"not make it",
        r"make it out", r"make it to your",
        r"won't get away", r"will not get away", r"not get away",
        r"won't leave", r"will not leave", r"not leave",
        r"can't escape", r"can not escape", r"not escape",
        r"regret leaving", r"regret it", r"stop you from leaving"
    ]
    if any(re.search(v, text_lower) for v in violence_keywords):
        return {"override": False} # ABORT IMMEDIATELY

    # 3. ML CONFIDENCE CHECK (The Safety Floor)
    if original_ml_score > 0.85:
        return {"override": False} # ABORT IMMEDIATELY

    # 3.5. SEMANTIC VETO (Phase 24)
    # If the Semantic Engine detects even mild danger proximity, don't dampen.
    if semantic_data:
        sem_score, _ = semantic_data
        if sem_score >= 0.20:
            return {"override": False}  # Semantic danger signal — ABORT

    # 4. APPLY DAMPENING (Safe Path)
    return {
        "override_tactic": "Accountability Check",
        "risk_modifier": 0.5, # Reduce risk by 50%
        "override": True
    }
