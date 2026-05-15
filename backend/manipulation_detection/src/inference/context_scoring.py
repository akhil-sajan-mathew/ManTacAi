"""
Context-aware scoring module for ManTacAi.

Computes initiator/reactor role attribution using ASYMMETRIC dampening:
  - Identifies the Primary Aggressor dynamically based on severity history.
  - The Primary Aggressor is LOCKED as an initiator and cannot be dampened.
  - DARVO tactics by the Primary Aggressor are ALWAYS attacks.
  - Victims using defensive or ambiguous tactics are classified as reactive
    and are eligible for dampening.
"""

import math

# Tactics that are inherently OFFENSIVE — using these IS aggression.
OFFENSIVE_TACTICS = {
    "deflection",
    "whataboutism",
    "guilt_tripping",
    "appeal_to_emotion",
    "gaslighting",
    "threatening_intimidation",
    "coercive_control",
    "love_bombing",
    "belittling_ridicule",
    "emotional_manipulation",
}

# Tactics that are AMBIGUOUS — context determines their meaning.
DEFENSIVE_TACTICS = {
    "passive_aggression",
    "stonewalling",
    "benign_venting",
    "healthy_conflict",
}

# Labels considered non-manipulative (safe labels)
SAFE_LABELS = {
    "neutral_conversation", "ethical_persuasion", "benign_venting",
    "healthy_conflict", "benign_affection", "neutral_logistics", "urgent_emergency"
}


def compute_attribution(
    isolated_preds,
    contextual_preds,
    preceding_risks,
    preceding_senders,
    preceding_labels,
    current_sender,
    is_echo=False,
    echo_similarity=0.0,
    is_primary_aggressor=False,
):
    """
    Determines whether the current message is an independent initiation of
    manipulation or a reactive echo of a preceding aggressor.

    ASYMMETRIC STRATEGY:
      1. Aggression is detected by TACTIC LABEL (is it in OFFENSIVE_TACTICS?),
         not just risk score.
      2. The dynamically identified Primary Aggressor is locked as an initiator.
      3. OFFENSIVE_TACTICS labels by the Primary Aggressor are always "initiator".
         If used by a victim, they are only "initiator" if they are undeniable 
         severe attacks (>0.75).
      4. Defensive/low-risk tactics after an attack default to "reactor".

    Args:
        isolated_preds (dict):         {label: prob} from classifying text alone
        contextual_preds (dict):       {label: prob} from classifying with context
        preceding_risks (list):        calculated risk scores of last N messages
        preceding_senders (list):      sender names of last N messages
        preceding_labels (list):       tactic labels of last N messages
        current_sender (str):          sender of the current message
        is_echo (bool):                True if semantic echo detection flagged as mirror
        echo_similarity (float):       cosine similarity score from echo check
        is_primary_aggressor (bool):   True if this sender is identified as the primary aggressor

    Returns:
        dict: {
            "role": "initiator" | "reactor" | "neutral",
            ...
        }
    """
    isolated_top_label = max(isolated_preds, key=isolated_preds.get) if isolated_preds else "neutral_conversation"
    isolated_top_score = isolated_preds.get(isolated_top_label, 0.0)

    contextual_top_label = max(contextual_preds, key=contextual_preds.get) if contextual_preds else "neutral_conversation"
    contextual_top_score = contextual_preds.get(contextual_top_label, 0.0)

    # ─── GATE 1: Safe label → neutral ───
    if isolated_top_label in SAFE_LABELS and isolated_top_score > 0.5:
        return {
            "role": "neutral",
            "dampening_factor": 1.0,
            "initiated_risk": 0.0,
            "reactive_risk": 0.0,
        }

    # ─── GATE 2: Very low confidence → neutral ───
    if isolated_top_score < 0.2:
        return {
            "role": "neutral",
            "dampening_factor": 1.0,
            "initiated_risk": 0.0,
            "reactive_risk": 0.0,
        }

    # ─── Check if a DIFFERENT speaker used an OFFENSIVE tactic recently ───
    other_speaker_was_aggressive = False
    distance_to_last_aggression = len(preceding_risks)

    for i in range(len(preceding_risks) - 1, -1, -1):
        sender = preceding_senders[i] if i < len(preceding_senders) else current_sender
        risk = preceding_risks[i] if i < len(preceding_risks) else 0.0
        label = preceding_labels[i] if i < len(preceding_labels) else "neutral_conversation"

        if sender != current_sender:
            label_is_offensive = label in OFFENSIVE_TACTICS
            risk_is_high = risk > 0.3

            if label_is_offensive or risk_is_high:
                other_speaker_was_aggressive = True
                distance_to_last_aggression = len(preceding_risks) - 1 - i
                break

    # ─── GATE 3: ASYMMETRIC — Established aggressors CANNOT become reactors ───
    # Only the dynamically identified Primary Aggressor gets locked into the Initiator role permanently.
    if is_primary_aggressor and isolated_top_label not in SAFE_LABELS:
        return {
            "role": "initiator",
            "dampening_factor": 1.0,
            "initiated_risk": isolated_top_score,
            "reactive_risk": 0.0,
        }

    # ─── GATE 4: DARVO Override — Offensive tactics by Aggressors are NEVER reactive ───
    # Primary Aggressors are always penalized for using DARVO.
    # Non-aggressors (victims) who use extreme DARVO (>0.75) are tagged as initiators
    # ONLY IF the context doesn't prove it's a desperate defense. If the context
    # dual-pass drastically reduces the score, it proves it was a reaction.
    if isolated_top_label in OFFENSIVE_TACTICS:
        if is_primary_aggressor:
            return {
                "role": "initiator",
                "dampening_factor": 1.0,
                "initiated_risk": isolated_top_score,
                "reactive_risk": 0.0,
            }
        
        context_reduces = contextual_top_score < isolated_top_score * 0.85
        if isolated_top_score > 0.75 and not context_reduces:
            return {
                "role": "initiator",
                "dampening_factor": 1.0,
                "initiated_risk": isolated_top_score,
                "reactive_risk": 0.0,
            }

    # ─── GATE 4.5: DARVO ESCALATION — Mild confrontation followed by manipulation ───
    # When a victim calmly raises an issue (low preceding risk) and the current
    # speaker responds with a clear offensive tactic, that's escalation, not defense.
    # Threshold set to 0.6 to avoid false positives in genuine mutual conflict.
    if other_speaker_was_aggressive and isolated_top_label in OFFENSIVE_TACTICS:
        preceding_nonzero = [r for r in preceding_risks[-3:] if r > 0]
        preceding_was_mild = all(r < 0.5 for r in preceding_nonzero) if preceding_nonzero else False
        current_is_clearly_manipulative = isolated_top_score > 0.6
        if preceding_was_mild and current_is_clearly_manipulative:
            return {
                "role": "initiator",
                "dampening_factor": 1.0,
                "initiated_risk": isolated_top_score,
                "reactive_risk": 0.0,
            }

    # ─── GATE 5: Defensive/ambiguous tactics after aggression → reactor ───
    if other_speaker_was_aggressive and isolated_top_label in DEFENSIVE_TACTICS:
        base_dampening = 0.2
        decay_rate = 3.0
        dampening = base_dampening + (1.0 - base_dampening) * (
            1.0 - math.exp(-distance_to_last_aggression / decay_rate)
        )
        dampening = max(dampening, 0.1)
        dampening = min(dampening, 1.0)

        return {
            "role": "reactor",
            "dampening_factor": dampening,
            "initiated_risk": 0.0,
            "reactive_risk": isolated_top_score * dampening,
        }

    if not other_speaker_was_aggressive:
        return {
            "role": "initiator",
            "dampening_factor": 1.0,
            "initiated_risk": isolated_top_score,
            "reactive_risk": 0.0,
        }

    # ─── The other speaker WAS aggressive, current label is NOT offensive/defensive ───
    # This is an ambiguous zone (or a victim using mild DARVO under 0.75).
    # Disambiguate with Context Reduction or Echo.
    context_reduces = contextual_top_score < isolated_top_score * 0.85
    semantic_echo = is_echo and echo_similarity > 0.70

    if context_reduces or semantic_echo:
        base_dampening = 0.2
        decay_rate = 3.0
        dampening = base_dampening + (1.0 - base_dampening) * (
            1.0 - math.exp(-distance_to_last_aggression / decay_rate)
        )

        if semantic_echo and echo_similarity > 0.85:
            dampening *= 0.7

        dampening = max(dampening, 0.1)
        dampening = min(dampening, 1.0)

        return {
            "role": "reactor",
            "dampening_factor": dampening,
            "initiated_risk": 0.0,
            "reactive_risk": isolated_top_score * dampening,
        }

    # ─── Final fallback: risk-level disambiguation ───
    if isolated_top_score < 0.3:
        return {
            "role": "neutral",
            "dampening_factor": 1.0,
            "initiated_risk": 0.0,
            "reactive_risk": 0.0,
        }

    if isolated_top_score < 0.5:
        # Victim defending mildly
        return {
            "role": "reactor",
            "dampening_factor": 0.5,
            "initiated_risk": 0.0,
            "reactive_risk": isolated_top_score * 0.5,
        }

    # Mutual toxicity (non-aggressor escalated to independent attack)
    return {
        "role": "initiator",
        "dampening_factor": 1.0,
        "initiated_risk": isolated_top_score,
        "reactive_risk": 0.0,
    }
