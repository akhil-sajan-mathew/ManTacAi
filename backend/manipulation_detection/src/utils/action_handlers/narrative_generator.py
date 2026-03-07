def generate_narrative_summary(metrics: dict) -> str:
    """
    Generates a human-readable summary of the detected manipulation tactics
    based on the computed risk metrics and tactic fingerprint.
    
    Args:
        metrics: The response dictionary from the analyze_chat endpoint.
                 Expected to contain 'risk_level', 'risk_score', 'darvo_score', 
                 'primary_pattern', and 'radar_chart_data'.
    """
    
    risk_level = str(metrics.get('risk_level', 'Safe')).upper()
    risk_score = float(metrics.get('risk_score', 0))
    darvo_score = float(metrics.get('darvo_score', 0))
    primary_pattern = str(metrics.get('primary_pattern', 'None'))
    radar_data = metrics.get('radar_chart_data', [])
    
    if risk_level == 'SAFE' and risk_score < 0.2:
        return "This conversation appears healthy and within normal boundaries. No significant manipulation patterns were detected."

    narrative = []
    
    # 1. Opening Statement based on overall risk
    if risk_level in ['CRITICAL', 'HIGH']:
        narrative.append(f"Analysis indicates a {risk_level} risk of psychological manipulation, characterized primarily by {primary_pattern}.")
    elif risk_level == 'MEDIUM':
        narrative.append(f"Moderate boundary testing and manipulative behaviors were detected, leaning towards {primary_pattern}.")
    else:
        narrative.append("Low-level friction exists, though it currently remains below the threshold for systemic abuse.")

    # 2. Specific Tactic Highlights (top 2 tactics > 0.3)
    tactics = sorted(radar_data, key=lambda x: float(x.get('A', 0)), reverse=True)
    significant_tactics = [t for t in tactics if float(t.get('A', 0)) >= 30.0]
    
    if len(significant_tactics) >= 2:
        t1, t2 = significant_tactics[0]['subject'], significant_tactics[1]['subject']
        narrative.append(f"The interaction is heavily driven by {t1} and {t2}.")
    elif len(significant_tactics) == 1:
        narrative.append(f"The interaction is primarily driven by {significant_tactics[0]['subject']}.")

    # 3. DARVO Context
    if darvo_score > 0.7:
        narrative.append("A severe DARVO (Deny, Attack, Reverse Victim/Offender) cycle is active, indicating a strong refusal of accountability.")
    elif darvo_score > 0.4:
        narrative.append("Elements of DARVO are present, suggesting an attempt to shift blame onto the victim.")

    return " ".join(narrative)
