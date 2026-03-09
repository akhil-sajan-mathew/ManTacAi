def generate_narrative_summary(metrics: dict) -> str:
    """
    Generates a comprehensive, structured narrative analysis of detected
    manipulation patterns based on the computed risk metrics.
    
    Returns a multi-section analysis formatted with markdown-style headers
    that the frontend renders as a structured document.
    """
    
    risk_level = str(metrics.get('risk_level', 'Safe')).upper()
    risk_score = float(metrics.get('risk_score', 0))
    darvo_score = float(metrics.get('darvo_score', 0))
    primary_pattern = str(metrics.get('primary_pattern', 'None'))
    radar_data = metrics.get('radar_chart_data', [])
    cycle_phase = str(metrics.get('cycle_phase', 'Normal'))
    
    if risk_level == 'SAFE' and risk_score < 0.2:
        return (
            "## Assessment: Safe\n\n"
            "This conversation appears healthy and within normal boundaries. "
            "No significant manipulation patterns were detected. The interaction "
            "shows respectful communication without coercive undertones.\n\n"
            "**Recommendation:** No action needed. Continue maintaining healthy boundaries."
        )

    sections = []
    
    # ─── SECTION 1: OVERALL RISK ASSESSMENT ───
    risk_pct = round(risk_score * 100, 1)
    if risk_level in ['CRITICAL', 'HIGH']:
        severity_desc = (
            f"The overall risk score of **{risk_pct}%** places this interaction in the "
            f"**{risk_level}** category. This indicates a pattern of sustained psychological "
            f"manipulation that may cause significant emotional harm to the recipient."
        )
    elif risk_level == 'MEDIUM':
        severity_desc = (
            f"The overall risk score of **{risk_pct}%** indicates **moderate** boundary-testing "
            f"behavior. While not yet at crisis level, these patterns can escalate over time "
            f"and should be monitored closely."
        )
    else:
        severity_desc = (
            f"The overall risk score of **{risk_pct}%** suggests **low-level** friction. "
            f"Some concerning language was detected, but it currently remains below the "
            f"threshold for systematic manipulation."
        )
    sections.append(f"## Risk Assessment\n\n{severity_desc}")
    
    # ─── SECTION 2: TACTIC BREAKDOWN ───
    tactic_descriptions = {
        'Gaslighting': (
            "Reality distortion tactics were detected — attempts to make the victim "
            "question their own memory, perception, or sanity. Common phrases include "
            "\"that never happened\" or \"you're imagining things.\""
        ),
        'Guilt': (
            "Guilt-tripping behavior was identified — leveraging obligation, sacrifice, "
            "or emotional debt to control the victim's actions. This often manifests as "
            "\"after everything I've done for you\" style rhetoric."
        ),
        'Threats': (
            "Threatening or intimidating language was detected — explicit or implied threats "
            "of harm, abandonment, or retaliation designed to enforce compliance through fear."
        ),
        'Silence': (
            "Stonewalling behavior was identified — deliberate withdrawal, refusal to engage, "
            "or the silent treatment used as a punishment mechanism to regain control."
        ),
        'Love Bomb': (
            "Love bombing patterns were detected — excessive flattery, affection, or grand gestures "
            "used strategically, often after conflict, to re-establish emotional dependency."
        ),
        'Deflection': (
            "Deflection tactics were identified — redirecting blame, changing the subject, "
            "or refusing to address legitimate concerns to avoid accountability."
        )
    }
    
    tactics = sorted(radar_data, key=lambda x: float(x.get('A', 0)), reverse=True)
    significant = [t for t in tactics if float(t.get('A', 0)) >= 20]
    
    if significant:
        tactic_lines = []
        for t in significant:
            name = t['subject']
            score = int(t.get('A', 0))
            desc = tactic_descriptions.get(name, f"Manipulative behavior categorized as {name}.")
            severity_label = "🔴 Critical" if score >= 70 else "🟠 High" if score >= 50 else "🟡 Moderate"
            tactic_lines.append(f"**{name}** — {severity_label} ({score}%)\n{desc}")
        
        sections.append("## Detected Tactics\n\n" + "\n\n".join(tactic_lines))
    else:
        sections.append(
            "## Detected Tactics\n\nNo individual tactic exceeded the significance threshold (20%). "
            "The interaction may contain subtle micro-aggressions that don't cluster into a single category."
        )
    
    # ─── SECTION 3: PRIMARY PATTERN ───
    if primary_pattern and primary_pattern not in ['None', 'NONE']:
        readable = primary_pattern.replace('_', ' ').title()
        sections.append(
            f"## Primary Pattern: {readable}\n\n"
            f"The dominant manipulation strategy identified is **{readable}**. "
            f"This pattern was the most consistently detected across the analyzed messages "
            f"and represents the core behavioral strategy being employed."
        )
    
    # ─── SECTION 4: DARVO ANALYSIS ───
    if darvo_score > 0.1:
        darvo_pct = round(darvo_score * 100, 1)
        if darvo_score > 0.7:
            darvo_text = (
                f"A **severe DARVO cycle** was detected (index: {darvo_pct}%). "
                f"DARVO (Deny, Attack, Reverse Victim & Offender) is a strong indicator "
                f"of manipulative intent. The subject is actively denying wrongdoing, "
                f"attacking the victim's credibility, and attempting to reverse the roles — "
                f"positioning themselves as the real victim."
            )
        elif darvo_score > 0.4:
            darvo_text = (
                f"**Partial DARVO elements** were detected (index: {darvo_pct}%). "
                f"At least two of the three DARVO components (Deny, Attack, Reverse) are "
                f"present, suggesting an emerging pattern of blame-shifting and accountability "
                f"avoidance."
            )
        else:
            darvo_text = (
                f"**Mild DARVO indicators** were noted (index: {darvo_pct}%). "
                f"Isolated denial or deflection tactics exist but haven't coalesced "
                f"into a full DARVO pattern yet."
            )
        sections.append(f"## DARVO Analysis\n\n{darvo_text}")
    
    # ─── SECTION 5: CYCLE PHASE ───
    phase_descriptions = {
        'NORMAL': "The conversation is currently in a **calm phase**. No active escalation detected.",
        'TENSION': (
            "The interaction is in the **tension-building phase**. Micro-aggressions "
            "and passive-aggressive behavior are accumulating. Historically, this phase "
            "precedes more overt aggression."
        ),
        'EXPLOSION': (
            "The conversation has entered the **explosion phase** — active, overt "
            "manipulation or aggression is occurring. This is the most dangerous phase "
            "of the abuse cycle."
        ),
        'HONEYMOON': (
            "A **honeymoon phase** has been detected — following a period of aggression, "
            "the subject is now displaying affection, apologies, or love-bombing. "
            "This is a well-documented pattern in cycles of abuse designed to regain trust."
        )
    }
    phase_upper = cycle_phase.upper().replace(' ', '_').split('/')[0].strip()
    phase_text = phase_descriptions.get(phase_upper, f"Current phase: {cycle_phase}.")
    sections.append(f"## Cycle Phase\n\n{phase_text}")
    
    # ─── SECTION 6: RECOMMENDATIONS ───
    if risk_level in ['CRITICAL', 'HIGH']:
        rec = (
            "- **Document everything** — save screenshots, timestamps, and context\n"
            "- **Seek professional support** — contact a domestic violence hotline or counselor\n"
            "- **Create a safety plan** — identify trusted contacts and safe locations\n"
            "- **Do not confront the manipulator** with this analysis, as it may trigger escalation"
        )
    elif risk_level == 'MEDIUM':
        rec = (
            "- **Monitor the pattern** — run follow-up analyses on future conversations\n"
            "- **Set clear boundaries** — address specific behaviors rather than character\n"
            "- **Confide in a trusted person** — share your concerns with someone you trust\n"
            "- **Consider professional guidance** if the pattern persists"
        )
    else:
        rec = (
            "- **Stay aware** — maintain healthy communication habits\n"
            "- **Trust your instincts** — if something feels wrong, it may be worth exploring\n"
            "- **Re-analyze** if new concerning messages emerge"
        )
    sections.append(f"## Recommendations\n\n{rec}")
    
    # ─── FOOTER ───
    sections.append(
        "---\n*This analysis is generated by ManTacAi's forensic engine and is intended "
        "for informational purposes only. It is not a substitute for professional psychological assessment.*"
    )
    
    return "\n\n".join(sections)

