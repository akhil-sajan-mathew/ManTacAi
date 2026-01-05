
import time
import json
import os

class CycleDetector:
    def __init__(self):
        self.state = "NORMAL"
        self.neutral_msg_count = 0
        self.last_high_risk_time = 0.0
        
        # Risk Categories
        self.ESCALATION_LABELS = {"physical_threat", "severe_stalking", "coercive_control"}
        self.TENSION_LABELS = {"passive_aggression", "deflection", "stonewalling", "guilt_tripping"}
        self.MANIPULATION_LABELS = {"love_bombing"}

    def update(self, label, score, timestamp=None):
        current_time = timestamp if timestamp is not None else time.time()
        
        # --- 1. PRIORITY CHECK: CRITICAL DANGER (Overrides everything) ---
        # SEVERITY INJECTION: High-Score Guilt/Belittling = Coercion
        if label == "guilt_tripping" and score > 0.85:
            self.state = "EXPLOSION"
            self.last_high_risk_time = current_time
            self.neutral_msg_count = 0
            return "CRITICAL_DANGER"

        if label in self.ESCALATION_LABELS:
            self.state = "EXPLOSION"
            self.last_high_risk_time = current_time # RESTART TIMER
            self.neutral_msg_count = 0
            return "CRITICAL_DANGER"

        # --- 2. TENSION LOGIC (Entry & Reinforcement) ---
        if label in self.TENSION_LABELS:
            # If Normal, enter Tension. If already Tension, JUST RESTART TIMER.
            if self.state == "NORMAL":
                self.state = "TENSION"
            
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
                    return "DE_ESCALATION_DETECTED"
            
            # Honeymoon is harder to exit. Needs more proof (e.g. 50 msgs).
            elif self.state == "HONEYMOON":
                 if self.neutral_msg_count > 50 and time_passed > 3600:
                    self.state = "NORMAL"
                    self.neutral_msg_count = 0

        return self.state

    def to_dict(self):
        return {
            "state": self.state,
            "neutral_msg_count": self.neutral_msg_count,
            "last_high_risk_time": self.last_high_risk_time
        }

    def from_dict(self, data):
        self.state = data.get("state", "NORMAL")
        self.neutral_msg_count = data.get("neutral_msg_count", 0)
        self.last_high_risk_time = data.get("last_high_risk_time", 0.0)


class ContextEngine:
    def __init__(self, persistence_file="context_state.json"):
        self.detector = CycleDetector()
        self.persistence_file = persistence_file
        self.history_buffer = [] # Visual history for debugging/UI
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

    def get_contextom(self):
        """Returns readable context for UI display"""
        return {
            "Phase": self.detector.state,
            "Last Incident": time.ctime(self.detector.last_high_risk_time) if self.detector.last_high_risk_time > 0 else "None",
            "Safe Msgs": self.detector.neutral_msg_count
        }

    def save_state(self):
        data = {
            "detector": self.detector.to_dict(),
            "history": self.history_buffer
        }
        try:
            with open(self.persistence_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Failed to save context state: {e}")

    def load_state(self):
        if not os.path.exists(self.persistence_file):
            return
            
        try:
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)
                self.detector.from_dict(data.get("detector", {}))
                self.history_buffer = data.get("history", [])
        except Exception as e:
            print(f"Failed to load context state: {e}")
