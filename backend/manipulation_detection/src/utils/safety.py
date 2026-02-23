
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

    # Define Tiers
    red_flags = [
        "There is a history of aggressive outbursts or property damage",
        "My physical safety or well-being has been threatened", 
        "This person isolates me from friends, family, or support systems",
        "I have been threatened with negative consequences (firing, eviction, reputation)"
    ]
    
    orange_flags = [
        "There is a power imbalance (e.g., boss/employee, financial dependent)",
        "This person monitors my activities, location, or communications",
        "This person frequently lies or denies things I know happened"
    ]
    
    yellow_flags = [
        "I feel afraid to speak my mind or say 'no'",
        "I feel constantly confused or like I'm 'walking on eggshells'",
        "I feel exhausted or drained after interacting with this person"
    ]

    # Check for presence
    has_red = any(item in checked_items for item in red_flags)
    has_orange = any(item in checked_items for item in orange_flags)
    has_yellow = any(item in checked_items for item in yellow_flags)
    
    recommendations = []

    # Tier 1: Red Flags (Critical Override)
    if has_red:
        return "Critical", 0.5, [
            "⚠️ SAFETY ALERT: Indicators of high-risk behavior or coercion detected.",
            "Prioritize your physical and professional safety immediately.",
            "Do not confront this person directly if you fear retaliation.",
            "Contact a support hotline or legal professional."
        ]
        
    # Tier 2: Orange Flags (Significant Context)
    if has_orange:
        return "Moderate", 0.2, [ # Returns "Moderate" so app.py doesn't override label, but adds +0.2 to score
            "Note: Structural imbalances or monitoring behaviors detected.",
            "These factors can significantly increase the impact of manipulation.",
            "Consider setting strict boundaries or seeking external advice."
        ]

    # Tier 3: Yellow Flags (Context Awareness)
    if has_yellow:
        return "Low", 0.1, [ # Adds +0.1 to score
            "Context: You reported feeling drained or confused.",
            "Trust your instincts—these feelings are valid warning signs.",
            "Focus on self-care and distinct boundary setting."
        ]

    return "Low", 0.0, []

def get_safety_resources():
    return """
    ### 🆘 Support Resources (India/Kerala)
    *   **Police / Emergency:** 112
    *   **Domestic Violence (Kerala 'Mitra'):** 181
    *   **Women's Helpline (All India):** 1091
    *   **Kerala Women's Commission:** 0471-2302590 / 0471-2300590
    *   **Mental Health ('Disha' Kerala):** 1056 (or 0471-2552056)
    *   **Legal Aid:** [Kerala State Legal Services Authority (KELSA)](https://kelsa.nic.in/)
    """

def get_dynamic_safety_plan(risk_level, detected_pattern, darvo_score):
    """
    Generates a dynamic safety plan based on risk level, tactic, and DARVO score.
    """
    plan_intro = f"### 🛡️ Your Personalized Safety Plan\n"
    plan_intro += f"**Risk Context:** {risk_level} Risk | **Pattern:** {detected_pattern}\n\n"
    
    sections = []

    # 1. IMMEDIATE SAFETY (Critical/High Override)
    if risk_level in ["High", "Critical"]:
        sections.append("""
#### 🚨 IMMEDIATE ACTION REQUIRED
*   **The 'Go-Bag':** Pack a bag with ID, cash, keys, medications, and copies of important docs. Hide it outside the home (e.g., at work or a friend's house).
*   **Safe Haven:** Identify a 24/7 public place you can run to immediately (Police Station, Hospital ER, 24-hr Store).
*   **Code Word:** Agree on a safe word/emoji with a trusted friend that means *"Call the police"* or *"Come pick me up now"*.
*   **Tech Safety:** Assume your phone is tracked. If possible, buy a cheap prepaid 'burner' phone and keep it hidden.
        """)

    # 2. TACTIC-SPECIFIC ADVICE
    tactic_advice = ""
    pattern_lower = detected_pattern.lower()

    if "gaslighting" in pattern_lower:
        tactic_advice = """
#### 📝 Countering Gaslighting (Reality Anchoring)
*   **Write It Down:** Immediately document events, dates, times, and exact quotes in a private, secure place.
*   **Do Not Argue Reality:** When they deny an event, say *"I know what I saw/heard"* once, then disengage.
*   **Trust Your Gut:** If you feel confused, it is likely because you are being manipulated, not because you are crazy.
        """
    elif "love bombing" in pattern_lower or "love_bombing" in pattern_lower:
        tactic_advice = """
#### ⏸️ Managing Love Bombing (The 'Pause' Button)
*   **Force Delays:** Intentionally slow down big decisions. Say *"I need time to think"* and stick to it.
*   **Maintain Autonomy:** Schedule fixed times to see YOUR friends and family without your partner.
*   **Watch for the Switch:** Love bombing often turns into devaluation the moment you set a boundary.
        """
    elif "stonewalling" in pattern_lower:
        tactic_advice = """
#### 🧱 Handling Stonewalling (The Silent Treatment)
*   **Disengage:** Do not beg for a response. Go do something that makes YOU happy.
*   **State Your Need:** Send one message: *"I'm ready to talk when you are willing to have a respectful conversation,"* then stop texting.
*   **Don't Chase:** Chasing them reinforces that their silence gives them power over your emotions.
        """
    elif "threatening" in pattern_lower or "intimidation" in pattern_lower:
         tactic_advice = """
#### 🛑 Dealing with Intimidation
*   **Physical Safety First:** If you feel physically unsafe, leave the room/house immediately.
*   **Do Not Escalate:** Do not mirror their aggression. Speak calmly or not at all.
*   **Report:** Threats of violence are illegal. Document them and report to authorities.
        """
    elif "guilt" in pattern_lower:
        tactic_advice = """
#### 🛡️ Deflecting Guilt Tripping
*   **Recognize the Tactic:** Ask yourself: *"Did I actually do something wrong, or are they just trying to make me feel bad?"*
*   **Refuse the Guilt:** You are allowed to say 'No' without being a bad person.
*   **Broken Record:** Repeat your boundary:        "I understand you are disappointed, but I cannot do that right now."*
        """
    elif "coercive" in pattern_lower or "control" in pattern_lower:
        tactic_advice = """
#### ⛓️ Regaining Autonomy (Coercive Control)
*   **Secure Documents:** Locate your passport, ID, and birth certificate. Keep them in a safe place outside the home.
*   **Financial Independence:** If possible, open a bank account in your name only, with paperless statements sent to a secure email.
*   **Reconnect:** Reach out to one friend or family member you've drifted from. You don't have to explain everything, just say "Hello."
        """
    
    if tactic_advice:
        sections.append(tactic_advice)

    # 3. DARVO DEFENSE
    if darvo_score > 0.6:
        sections.append("""
#### 🔄 Anti-Reversal Strategy (DARVO)
*   **Don't J.A.D.E.:** Do not **J**ustify, **A**rgue, **D**efend, or **E**xplain your position.
*   **Stay on Topic:** If they switch to playing the victim, say: *"I hear that you are upset, but right now we are discussing [Original Issue]."*
*   **Walk Away:** If they continue to attack you to distract from their actions, end the conversation.
        """)

    # 4. GENERAL FOUNDATIONS
    if risk_level in ["Low", "Medium"] or len(sections) < 1:
        sections.append("""
#### 🏗️ Foundation & Boundaries
*   **Digital Hygiene:** Change passwords, enable 2FA, and check 'active sessions' on your accounts.
*   **Support Network:** Identify 2 trusted people you can speak to openly. Silence protects the manipulator.
*   **Grey Rock Method:** If you must interact, be as boring as a grey rock. Give short, factual answers without emotion.
        """)
    
    # 5. RESOURCES
    sections.append("""
---
#### 🆘 Expert Support (India/Kerala)
*   **Police / Emergency:** 112
*   **Domestic Violence (Kerala 'Mitra'):** 181
*   **Women's Helpline:** 1091
*   **Mental Health ('Disha'):** 1056
*   **Legal Aid:** [KELSA](https://kelsa.nic.in/) or Search "Legal Aid Kerala"
    """)

    return plan_intro + "\n".join(sections)
