# ManTacAi: Advanced Manipulation & Abuse Detection Tool

**ManTacAi** is an AI-powered forensic tool designed to detect subtle signs of domestic violence, coercive control, and psychological manipulation in text conversations. 

Unlike standard sentiment analyzers, ManTacAi is trained to identify the **"Wolf in Sheep's Clothing"**—manipulation disguised as care, love, or rational concern. It uses a **Hybrid Architecture** combining a fine-tuned Transformer model (V8) with a rule-based Context Engine to track the Cycle of Abuse.

![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Model](https://img.shields.io/badge/Model-V8%20Platinum-blue)
![Classes](https://img.shields.io/badge/Classes-18%20Types-orange)

---

## 🔍 The 18-Point Taxonomy (Class Labels)
ManTacAi classifies every message into one of **18 distinct categories**, ranging from critical safety threats to healthy communication.

| Risk Level | Label | Description | Example |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | `urgent_emergency` | Life-threatening situations. Triggers immediate UI Red Alert. | *"Call 911", "He has a knife", "Help me"* |
| **CRITICAL** | `threatening_intimidation` | Overt threats of harm to self, partner, or pets. | *"If you leave, you'll never see the kids again."* |
| **EXTREME** | `coercive_control` | Controlling behavior masked as protection or love. | *"I deleted your social media because it makes you anxious."* |
| **HIGH** | `gaslighting` | Denying reality to make the victim doubt their sanity. | *"I never said that. You're imagining things again."* |
| **HIGH** | `belittling_ridicule` | Insults designed to lower self-esteem. | *"You're so stupid, nobody else would put up with you."* |
| **MODERATE** | `stonewalling` | Refusing to communicate to punish the partner. | *"..." (Silence for days)* |
| **MODERATE** | `guilt_tripping` | Using guilt to manipulate behavior. | *"I guess my feelings don't matter to you."* |
| **MODERATE** | `love_bombing` | Excessive affection used to hook or reclaim a victim. | *"We are twin flames. You are my destiny."* |
| **MODERATE** | `deflection` | Shifting blame away from oneself. | *"I wouldn't get angry if you didn't provoke me."* |
| **LOW** | `passive_aggression` | Indirect expression of hostility. | *"Fine. Do whatever you want like you always do."* |
| **LOW** | `whataboutism` | Counter-accusing to avoid accountability. | *"What about that time YOU forgot to call?"* |
| **LOW** | `appeal_to_emotion` | Manipulating emotions without logic. | *"If you loved me, you would do this."* |
| **SAFE** | `ethical_persuasion` | Healthy negotiation and "I" statements. | *"I feel hurt when you yell. Can we talk calmly?"* |
| **SAFE** | `healthy_conflict` | Disagreement without toxicity. | *"I disagree with your choice, but I respect it."* |
| **SAFE** | `benign_venting` | Frustration directed at external factors (work, games). | *"I hate this stupid boss! He's an idiot."* |
| **SAFE** | `benign_affection` | Normal expressions of love. | *"I miss you, can't wait to see you."* |
| **SAFE** | `neutral_logistics` | Planning and coordination. | *"Did you pick up the milk?"* |
| **SAFE** | `neutral_conversation` | General chit-chat. | *"The movie was okay."* |

---

## 🌟 Powerful App Capabilities

### 🎯 Forensic Target Selection
ManTacAi isn't just a classified; it's a **forensic tool**.
*   **Sender Filtering:** You can input a chat log containing two people (e.g., "Alex: ...", "Sarah: ...") and tell the AI to **only analyze "Alex"**. It intelligently parses the log, ignores the other person, and builds a psychological profile of the suspect alone.
*   **Smart Log Parsing:** It automatically strips timestamps (e.g., `[14:30]`) and metadata to focus purely on linguistic patterns.

### 🧠 The Context Window (Memory)
Unlike simple chatbots, ManTacAi has **Long-Term Memory** via its Context Engine.
*   **Sequential Analysis:** It reads messages in order. A "Love Bombing" message sent *after* an "Explosion" is flagged differently than one sent at the start.
*   **Persistence Tracking:** It tracks how long a tactic (like Stonewalling) has been happening. If silence persists for multiple turns, the risk score escalates.

### 🛡️ Safety & Reporting
*   **Dynamic Safety Plan:** Based on the specific abuse detected (e.g., Financial Control vs. Physical Threat), the app generates a custom safety checklist.
*   **Professional Reporting:** Click "Download Report" to generate a **Forensic .docx File**. This report summarizes risk levels, primary tactics, and DARVO scores, suitable for sharing with therapists or legal counsel.
*   **Benign Filter:** A "Common Sense" layer that filters out frustration about work, traffic, or video games, preventing false alarms.

---

## 🧠 Architecture: How It Works

The system processes every message through **Three Layers of Logic**:

### Layer 1: The Transformer (V8 Model)
*   **Engine:** `DistilRoBERTa` (Fine-Tuned)
*   **Function:** Reads raw text and outputs probability scores for all 18 classes.
*   **Result:** "Message A = 85% Gaslighting, 10% Deflection".

### Layer 2: The Context Engine (State Machine)
Abuse happens in a cycle. The engine tracks a hidden "Cycle State" across the conversation:
1.  **Tension Building:** Sustained passive aggression or stonewalling.
2.  **Explosion:** Threats or ridicule.
3.  **Honeymoon:** Sudden love bombing *after* an explosion.
*   **Logic:** If the cycle is in `Honeymoon` phase, the risk score of "sweet" messages is artificially raised (Threshold > 55%) because they are likely manipulative in this context.

### Layer 3: The Safety Guardrails
*   **Emergency Override:** If `urgent_emergency` is detected, the UI locks to **RED**, ignoring all other logic.
*   **DARVO Calculator:** Scores the presence of Deny, Attack, and Reverse Victim tactics.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   Python 3.10+
*   Git (with **Git LFS** enabled)

### 2. Clone Repository
```bash
git lfs install
git clone https://github.com/your-username/ManTacAi.git
cd ManTacAi
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
*   Access the Web UI at: `http://127.0.0.1:7860`

---

## 📂 Project Structure

*   `app.py`: The main Gradio application and UI logic.
*   `manipulation_detection/`: Core Python package.
    *   `src/inference/`: Model loading (`model.py`) and scoring logic (`scoring.py`).
    *   `src/utils/`:
        *   `context_engine.py`: The Cycle of Abuse state machine.
        *   `safety.py`: Emergency planning logic.
        *   `report.py`: Text report generator.
*   `manipulation_tactic_detector_model/`: The V8 model weights (`.safetensors`).
*   `dataset_augmented/`: The high-quality synthetic training data (JSON).
*   `scripts/`: Tools used to generate data and train the model.

---

## ⚠️ Limitations
*   **Short Text Paranoia:** The model can sometimes flag ultra-short texts (e.g., "Ok", "Fine") as *Stonewalling* because it lacks tone cues.
*   **Anxiety vs Abuse:** Phrases like "Am I crazy?" can trigger *Gaslighting* flags because the model associates the word "crazy" with abuse.
*   **Usage:** This tool is for **educational and forensic analysis**. It is not a replacement for human judgment or professional therapy.

---

## 📄 License
This project is open-source.