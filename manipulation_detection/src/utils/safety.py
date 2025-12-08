
SAFETY_CHECKLIST_ITEMS = [
    "There is a power imbalance (e.g., boss/employee, financial dependent)",
    "I feel afraid to speak my mind or say 'no'",
    "This person isolates me from friends, family, or support systems",
    "There is a history of aggressive outbursts or property damage",
    "I feel constantly confused or like I'm 'walking on eggshells'",
    "This person monitors my activities, location, or communications",
    "I have been threatened with negative consequences (firing, eviction, reputation)",
    "My physical safety or well-being has been threatened",
    "This person frequently lies or denies things I know happened",
    "I feel exhausted or drained after interacting with this person"
]

def evaluate_safety_risk(checked_items):
    """
    Evaluates safety risk based on checked checklist items.
    Returns: (risk_level, risk_score_modifier, safety_recommendations)
    """
    if not checked_items:
        return "Low", 0.0, []

    # High Risk Items (Immediate Critical/High Risk)
    high_risk_triggers = [
        "This person isolates me from friends, family, or support systems",
        "There is a history of aggressive outbursts or property damage",
        "This person monitors my activities, location, or communications",
        "I have been threatened with negative consequences (firing, eviction, reputation)",
        "My physical safety or well-being has been threatened"
    ]
    
    # Check if any high risk items are present
    high_risk_present = any(item in checked_items for item in high_risk_triggers)
    
    risk_score_modifier = 0.0
    recommendations = []

    if high_risk_present:
        return "Critical", 0.4, [
            "⚠️ SAFETY ALERT: Indicators of high-risk behavior or coercion detected.",
            "Prioritize your physical and professional safety immediately.",
            "Do not confront this person directly if you fear retaliation.",
            "Document all interactions and secure your devices/accounts if possible.",
            "Consider contacting a legal professional, HR representative, or support hotline."
        ]
    
    # If not high risk but items are checked -> Elevated Risk
    return "High", 0.2, [
        "Caution: The context suggests a toxic or manipulative dynamic.",
        "Trust your instincts—feeling afraid or confused is a valid warning sign.",
        "Set firm boundaries and try to limit engagement where possible.",
        "Seek support from a trusted third party (friend, mentor, or therapist)."
    ]

def get_safety_resources():
    return """
    ### 🆘 Support Resources
    *   **General/Domestic Violence:** [The Hotline](https://www.thehotline.org/) (1-800-799-7233)
    *   **Workplace Harassment:** [EEOC](https://www.eeoc.gov/) or your local labor board.
    *   **Mental Health:** [988 Lifeline](https://988lifeline.org/)
    """
