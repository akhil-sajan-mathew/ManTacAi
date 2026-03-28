
import time
import json
import os

from utils.encryption import save_encrypted, load_encrypted

class SpeakerProfile:
    """Tracks cumulative behavioral metrics for a single speaker."""
    def __init__(self, name):
        self.name = name
        self.message_count = 0
        self.high_risk_count = 0       # messages with risk > 0.5
        self.initiation_count = 0      # times this speaker initiated aggression
        self.reaction_count = 0        # times flagged as reactor
        self.neutral_count = 0         # times flagged as neutral
        self.tactic_set = set()        # unique tactic labels used
        self.total_risk_sum = 0.0      # for computing average risk
        self.first_strike_count = 0    # first toxic msg in an exchange window

    @property
    def initiation_ratio(self):
        """Fraction of flagged messages that were initiations (not reactions)."""
        total = self.initiation_count + self.reaction_count
        return self.initiation_count / total if total > 0 else 0.0

    @property
    def tactic_diversity(self):
        """Number of distinct manipulation tactics this speaker has used."""
        return len(self.tactic_set)

    @property
    def avg_risk(self):
        """Average risk score across all this speaker's messages."""
        return self.total_risk_sum / self.message_count if self.message_count > 0 else 0.0

    def update(self, label, risk_score, role):
        """Update profile with a new classified message."""
        self.message_count += 1
        self.total_risk_sum += risk_score

        if risk_score > 0.5:
            self.high_risk_count += 1

        safe_labels = {
            "neutral_conversation", "ethical_persuasion", "benign_venting",
            "healthy_conflict", "benign_affection", "neutral_logistics", "urgent_emergency"
        }
        if label not in safe_labels:
            self.tactic_set.add(label)

        if role == "initiator":
            self.initiation_count += 1
        elif role == "reactor":
            self.reaction_count += 1
        else:
            self.neutral_count += 1

    def to_dict(self):
        return {
            "name": self.name,
            "message_count": self.message_count,
            "high_risk_count": self.high_risk_count,
            "initiation_count": self.initiation_count,
            "reaction_count": self.reaction_count,
            "neutral_count": self.neutral_count,
            "tactic_set": list(self.tactic_set),
            "total_risk_sum": self.total_risk_sum,
            "first_strike_count": self.first_strike_count,
            "initiation_ratio": self.initiation_ratio,
            "tactic_diversity": self.tactic_diversity,
            "avg_risk": self.avg_risk
        }

    @classmethod
    def from_dict(cls, data):
        profile = cls(data.get("name", "Unknown"))
        profile.message_count = data.get("message_count", 0)
        profile.high_risk_count = data.get("high_risk_count", 0)
        profile.initiation_count = data.get("initiation_count", 0)
        profile.reaction_count = data.get("reaction_count", 0)
        profile.neutral_count = data.get("neutral_count", 0)
        profile.tactic_set = set(data.get("tactic_set", []))
        profile.total_risk_sum = data.get("total_risk_sum", 0.0)
        profile.first_strike_count = data.get("first_strike_count", 0)
        return profile


class CycleDetector:
    def __init__(self):
        self.state = "NORMAL"
        self.neutral_msg_count = 0
        self.last_high_risk_time = 0.0
        self.tension_count = 0 # Slow burn counter
        
        # Risk Categories (Mapped to 18-Label Model)
        self.ESCALATION_LABELS = {"threatening_intimidation", "coercive_control", "urgent_emergency"}
        self.TENSION_LABELS = {"gaslighting", "stonewalling", "belittling_ridicule", "passive_aggression"}
        self.MANIPULATION_LABELS = {"love_bombing", "guilt_tripping", "deflection", "whataboutism", "appeal_to_emotion"}

    def update(self, label, score, timestamp=None):
        current_time = timestamp if timestamp is not None else time.time()
        
        # --- 0. CIRCUIT BREAKER (Phase 19 Future-Proofing) ---
        # If Risk is CRITICAL (>0.85), Force EXPLOSION immediately.
        # This overrides all previous state (Honeymoon, Normal, etc).
        if score > 0.85:
            self.state = "EXPLOSION"
            self.last_high_risk_time = current_time
            self.neutral_msg_count = 0
            self.tension_count = 0
            return "CRITICAL_DANGER"

        # --- 1. PRIORITY CHECK: CRITICAL DANGER (Overrides everything) ---
        # SEVERITY INJECTION: High-Score Guilt/Belittling = Coercion
        # (Redundant due to Circuit Breaker but kept for label-specific logic if needed)
        if label == "guilt_tripping" and score > 0.85:
            self.state = "EXPLOSION"
            self.last_high_risk_time = current_time
            self.neutral_msg_count = 0
            self.tension_count = 0
            return "CRITICAL_DANGER"

        if label in self.ESCALATION_LABELS:
            self.state = "EXPLOSION"
            self.last_high_risk_time = current_time # RESTART TIMER
            self.neutral_msg_count = 0
            self.tension_count = 0
            return "CRITICAL_DANGER"

        # --- 2. TENSION LOGIC (Entry & Reinforcement) ---
        if label in self.TENSION_LABELS:
            # FIX: HONEYMOON BREAKER
            # If in Honeymoon, ANY tension breaks the illusion immediately.
            if self.state == "HONEYMOON":
                 self.state = "TENSION"

            # If Normal, enter Tension. If already Tension, JUST RESTART TIMER.
            if self.state == "NORMAL":
                self.state = "TENSION"
            
            # SLOW BURN LOGIC: Trace repeated micro-aggressions
            self.tension_count += 1
            if self.tension_count > 3:
                 self.state = "TENSION" # Force Tension/Devaluation
            
            # CRITICAL FIX: Always update time on bad behavior, regardless of state
            self.last_high_risk_time = current_time 
            self.neutral_msg_count = 0
            return self.state

        # --- 3. CYCLE DETECTION (Honeymoon Phase) ---
        if self.state == "EXPLOSION" and label in self.MANIPULATION_LABELS:
            self.state = "HONEYMOON"
            self.last_high_risk_time = current_time
            return "CYCLE_CONFIRMED"

        # --- 4. THE HYBRID GATE (The Safe Reset) ---
        # We define "Safe" strictly. Can add "appreciation" or "ethical_persuasion" here.
        if label in ["neutral_conversation", "ethical_persuasion", "benign_venting", "healthy_conflict", "benign_affection", "neutral_logistics", "urgent_emergency"]:
            self.neutral_msg_count += 1
            
            time_passed = current_time - self.last_high_risk_time
            
            # Logic: Need BOTH Time (1 hour) AND Volume (20 msgs) to believe it's safe.
            if self.state in ["TENSION", "EXPLOSION"]:
                if self.neutral_msg_count > 20 and time_passed > 3600:
                    self.state = "NORMAL"
                    self.neutral_msg_count = 0
                    self.tension_count = 0
                    return "DE_ESCALATION_DETECTED"
            
            # Honeymoon is harder to exit. Needs more proof (e.g. 50 msgs).
            elif self.state == "HONEYMOON":
                 if self.neutral_msg_count > 50 and time_passed > 3600:
                    self.state = "NORMAL"
                    self.neutral_msg_count = 0
                    self.tension_count = 0

        return self.state

    def to_dict(self):
        return {
            "state": self.state,
            "neutral_msg_count": self.neutral_msg_count,
            "last_high_risk_time": self.last_high_risk_time,
            "tension_count": self.tension_count
        }

    def from_dict(self, data):
        self.state = data.get("state", "NORMAL")
        self.neutral_msg_count = data.get("neutral_msg_count", 0)
        self.last_high_risk_time = data.get("last_high_risk_time", 0.0)
        self.tension_count = data.get("tension_count", 0)


class ContextEngine:
    def __init__(self, persistence_file="context_state.json"):
        self.detector = CycleDetector()
        self.persistence_file = persistence_file
        self.history_buffer = [] # Visual history for debugging/UI
        self.speaker_profiles = {}  # {sender_name: SpeakerProfile}
        self.load_state()

    def add_event(self, text, label, score, timestamp=None):
        # 1. Update State Machine
        status_update = self.detector.update(label, score, timestamp)
        
        # 2. Add to local buffer (last 50 messages)
        self.history_buffer.append({
            "text": text[:50] + "..." if len(text) > 50 else text,
            "label": label,
            "score": score,
            "timestamp": timestamp if timestamp is not None else time.time()
        })
        if len(self.history_buffer) > 50:
            self.history_buffer.pop(0)
            
        # 3. Save Logic (Persistence)
        self.save_state()
        
        return {
            "current_state": self.detector.state,
            "status_update": status_update,
            "risk_score": score # Pass through raw score for reference
        }

    def update_speaker_profile(self, sender_name, label, risk_score, role="neutral"):
        """Update or create a speaker profile with classification results."""
        if sender_name not in self.speaker_profiles:
            self.speaker_profiles[sender_name] = SpeakerProfile(sender_name)
        self.speaker_profiles[sender_name].update(label, risk_score, role)

    def get_speaker_profiles(self):
        """Returns all speaker profiles as a dict of dicts."""
        return {name: profile.to_dict() for name, profile in self.speaker_profiles.items()}

    def get_contextom(self):
        """Returns readable context for UI display"""
        return {
            "Phase": self.detector.state,
            "Last Incident": time.ctime(self.detector.last_high_risk_time) if self.detector.last_high_risk_time > 0 else "None",
            "Safe Msgs": self.detector.neutral_msg_count
        }

    def save_state(self):
        if not self.persistence_file: return
        
        data = {
            "detector": self.detector.to_dict(),
            "history": self.history_buffer,
            "speaker_profiles": {name: p.to_dict() for name, p in self.speaker_profiles.items()}
        }
        try:
            save_encrypted(self.persistence_file, data)
        except Exception as e:
            print(f"Failed to save context state: {e}")

    def load_state(self):
        if not self.persistence_file: return
        
        if not os.path.exists(self.persistence_file):
            return
            
        try:
            data = load_encrypted(self.persistence_file)
            self.detector.from_dict(data.get("detector", {}))
            self.history_buffer = data.get("history", [])
            # Restore speaker profiles
            for name, pdata in data.get("speaker_profiles", {}).items():
                self.speaker_profiles[name] = SpeakerProfile.from_dict(pdata)
        except Exception as e:
            print(f"Failed to load context state: {e}")

    def reset(self):
        """Hard reset of the context engine"""
        self.detector = CycleDetector()
        self.history_buffer = []
        self.speaker_profiles = {}
        if self.persistence_file and os.path.exists(self.persistence_file):
            try:
                os.remove(self.persistence_file)
            except Exception as e:
                print(f"Failed to delete persistence file: {e}")
        self.save_state()
