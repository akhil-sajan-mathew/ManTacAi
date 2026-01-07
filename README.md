# ManTacAi: Advanced Manipulation & Abuse Detection Tool

**ManTacAi** is an AI-powered forensic tool designed to detect subtle signs of domestic violence, coercive control, and psychological manipulation in text conversations. 

Unlike standard sentiment analyzers, ManTacAi is trained to identify the **"Wolf in Sheep's Clothing"**—manipulation disguised as care, love, or rational concern.

![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Model](https://img.shields.io/badge/Model-V8%20Platinum-blue)
![Safety](https://img.shields.io/badge/Safety-First-red)

---

## 🚀 Key Features

### 1. The V8 "Platinum" Model (18 Classes)
Most models only see "Negative Sentiment." ManTacAi detects **18 specific tactic classes**, including:
*   **Coercive Control:** *"I deleted your apps to protect you."*
*   **Love Bombing:** *"We are twin flames, nobody else matters."*
*   **Gaslighting:** *"You are imagining things, I never said that."*
*   **DARVO:** (Deny, Attack, Reverse Victim & Offender).
*   **Urgent Emergency:** *"Call 911"*, *"Help"* (Instantly flagged).
*   **Ethical Persuasion:** Distinguishes healthy negotiation from manipulation.

### 2. Contextual "Cycle of Abuse" Engine
The AI doesn't just look at single messages. It tracks the **Cycle of Abuse**:
1.  **Tension Building:** Passive aggression, stonewalling.
2.  **Explosion:** Threats, insults, humiliation.
3.  **Honeymoon:** Apologies, gifts, love bombing.
*   **Thresholds:** It adjusts sensitivity based on the active phase. A "joke" during the *Explosion* phase is treated differently than in the *Honeymoon* phase.

### 3. Safety-First Architecture
*   **Emergency Override:** If the model detects a 911/Life-Threatening situation, the UI immediately locks into **RED ALERT** mode, regardless of the prompt.
*   **Benign Filter:** Filters out tech support complaints ("My computer is broken") from actual threats, reducing false positives.
*   **Safety Plan:** Generates dynamic, actionable safety advice based on the specific type of abuse detected.

### 4. Forensic Reporting
*   **DARVO Score:** Calculates the likelihood of "Reverse Victim" tactics.
*   **Exportable Report:** Generates a professional `.docx` report summarizing the analysis for therapists or legal counsel.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   Git (with **Git LFS** support for big model files)

### Step 1: Clone the Repository
This repository uses Git LFS to store the ~500MB model weights.
```bash
git lfs install
git clone https://github.com/your-username/ManTacAi.git
cd ManTacAi
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚦 Usage

1.  **Launch the App**:
    ```bash
    python app.py
    ```
2.  **Access UI**: Open your browser at `http://127.0.0.1:7860`.
3.  **Analyze**:
    *   Paste chat logs into the input boxes.
    *   (Optional) Check the "Safety Checklist" items.
    *   Click **Analyze**.
4.  **Review**:
    *   See the **Risk Score** (Low/Moderate/High/Critical).
    *   Read the **Cycle Phase** analysis.
    *   Download the **Word Report**.

---

## 🧠 How It Works (Architecture)

The system is a **Hybrid AI**:

1.  **Layer 1: The Transformer (V8 Model)**
    *   A Fine-Tuned `DistilRoBERTa` model.
    *   Input: Raw text.
    *   Output: Probabilities for 18 tactic classes.

2.  **Layer 2: The Context Engine (Rule-Based)**
    *   Input: Model predictions + Timestamp + History.
    *   Logic:
        *   *State Machine:* Tracks persistence of abuse (e.g., if "Guilt Tripping" persists for >3 turns, enter "Tension" phase).
        *   *Overrides:* If Phase = Honeymoon AND Score > 55%, force **High Risk**.

3.  **Layer 3: The Safety & Scoring Layer**
    *   Calculates `DARVO` score.
    *   Applies `Emergency` patches.
    *   Generates the final Verdict.

---

## ⚠️ Disclaimer
**ManTacAi is an educational and support tool, not a diagnostic medical device.** 
Always prioritize physical safety. If you are in danger, contact local emergency services immediately.

---
*ManTacAi V8 - Built for Safety.*