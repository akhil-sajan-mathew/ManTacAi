# 🧠 ManTacAi: Forensic Manipulation & Abuse Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.0-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

A state-of-the-art forensic AI system designed to detect, classify, and analyze psychological manipulation in textual communication. ManTacAi combines deep learning (DistilRoBERTa) with a custom "Context Engine" to identify 18 distinct abuse tactics, improved by a safety-first "Cycle of Abuse" tracking system.

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [AI Models](#-ai-models)
- [Dataset](#-dataset)
- [Technical Specifications](#-technical-specifications)
- [Installation](#-installation)
- [Usage](#-usage)
- [Performance Metrics](#-performance-metrics)
- [File Structure](#-file-structure)
- [Disclaimer](#-disclaimer)

---

## ✨ Features

### Core Capabilities
- **🔍 18-Class Manipulation Detection**: Identifies tactics ranging from *Gaslighting* and *Love Bombing* to *Coercive Control* and *Stonewalling*.
- **🧠 Context-Aware Cycle Tracking**: Tracks the "Cycle of Abuse" (Tension Building → Explosion → Honeymoon) to flag patterns that single messages miss.
- **🚨 Emergency Safety Guardrails**: Hard-coded overrides for threats of self-harm or violence (`urgent_emergency` class) with 98.5% detection reliability.
- **📄 Forensic Reporting**: Generates downloadable Word/PDF reports with risk cards, timelines, and "DARVO" scores for evidence documentation.
- **⚡ Real-time Analysis**: Processes conversation logs locally on-device for maximum privacy.

### Supported Inputs
- **Text Logs**: Direct chat export analysis.
- **Interactive UI**: Real-time typing analysis via Gradio Interface.

---

## 🏗️ System Architecture

ManTacAi uses a **Hybrid 3-Layer Logic** system to balance raw AI power with human-defined safety and linguistic precision.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ManTacAi Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌────────────────────┐    ┌────────────────────────┐   │
│  │ Input Text   │───▶│ Layer 1: NLP Core  │───▶│ Layer 2: The Brain     │   │
│  │ (Chat Logs)  │    │ (spaCy + Regex)    │    │ (Deep Learning Model)  │   │
│  └──────────────┘    │                    │    │                        │   │
│                      │ • Tokenize/Lemma   │    │ • DistilRoBERTa V8     │   │
│                      │ • "Smart" Regex    │    │ • 18-Class Probabilities│  │
│                      │ • Keyword Filter   │    │ • Semantic Vectors     │   │
│                      └──────────────┬─────┘    └──────────┬─────────────┘   │
│                                     │                     │                 │
│                                     ▼                     ▼                 │
│  ┌──────────────┐    ┌────────────────────┐    ┌────────────────────────┐   │
│  │ Layer 4:     │◀───│ Layer 3: Context   │◀───│ Layer 2.5: Semantic    │   │
│  │ Safety V2    │    │ Engine (Memory)    │    │ Threat Engine          │   │
│  │ (Circuit     │    │                    │    │                        │   │
│  │  Breaker)    │    │ • Cycle Tracking   │    │ • Cosine Similarity    │   │
│  │              │    │ • Anti-Dampening   │    │ • Slang/Code Detection │   │
│  └──────┬───────┘    └────────────────────┘    └────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          Forensic Output                             │   │
│  │  • Verdict: "CRITICAL RISK - Direct Threat Detected"                 │   │
│  │  • Evidence: "I will ending you" (Lemma Match: "end")                │   │
│  │  • Cycle State: "Explosion Phase" (Circuit Breaker Active)           │   │
│  │  • Report: generated_report.docx                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 AI Models

### V8 Manipulation Detector (Fine-Tuned Transformer)
**Purpose**: Classify specific psychological tactics in conversational text.
**Architecture**: DistilRoBERTa-Base (Fine-tuned)

### Semantic Threat Engine (Vector Embeddings)
**Purpose**: Detect **Conceptual Threats** (slang, coded language) that bypass keyword filters.
**Mechanism**:
*   Extracts 768-dimensional vector from the Transformer's hidden state.
*   Calculates **Cosine Similarity** against a pre-computed "Direct Threat" centroid.
*   **Trigger**: >0.90 similarity flags as CRITICAL immediately.

### NLP Normalization Engine (spaCy)
**Purpose**: Smart pattern matching using **Lemmatization**.
**Benefit**: Reduces 1,000+ regex permutations (e.g., *kill, killing, killed*) to a single root form (*kill*), improving recall by ~40%.

---

## 📊 Dataset

### Source
**ManTacAi Synthetic & Augmented V8 Dataset** - A curated, balanced dataset specifically designed for forensic linguistics.

### Statistics

| Category | Description | Performance (F1) |
|----------|-------------|------------------|
| **High Risk** | Gaslighting, Coercive Control, Threats | **98.9%** |
| **Subtle** | Passive Aggression, Guilt Tripping | **92%** |
| **Safety** | Emergency / Self-Harm | **98.5%** |
| **Healthy** | Ethical Persuasion, Neutral, Benign | **99%** |
| **Total** | **8,000+** Curated Examples | - |

**Note on Data:** The dataset is heavily augmented with "Boring/Neutral" examples to prevent the AI from becoming paranoid (false positive reduction).

---

## ⚙️ Technical Specifications

### Risk Assessment Logic (Safety Core V2)
The raw probability is not enough. We calculate a weighted **Risk Score**:

```python
Risk Score = (Max_Prob * Severity_Weight * Dampening_Factor)
```

1.  **Anti-Dampening (Trojan Horse Defense)**: High-risk heuristics (e.g., Financial Abuse) are *dampened* (0.1x) in safe contexts but **unlocked** (1.0x) if violence is detected.
2.  **Safety Floor**: If the AI model predicts >85% confidence for a threat, heuristics CANNOT lower the score.

### Context Engine (Cycle of Abuse)
The system maintains a rolling state window to detect the **Cycle of Abuse**:
1.  **Tension Building**: Rising frequency of Passive Aggression/Stonewalling.
2.  **Explosion**: High confidence Threats or Belligerence.
3.  **Honeymoon**: Sudden shift to Love Bombing/Apologies after an Explosion.

**Circuit Breaker Protocol**:
*   If ANY message exceeds **0.85 Risk Score** (Critical), the phase is **IMMEDIATELY** forced to "Explosion", overriding any "Honeymoon" attempt. This prevents abusers from "resetting" the cycle instantly.

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Git

### Setup
```bash
# 1. Clone repository
git clone https://github.com/akhil-sajan-mathew/ManTacAi
cd ManTacAi

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy Model (Required for Phase 22)
python -m spacy download en_core_web_sm

# 4. (Optional) Install LFS for Model Weights
git lfs install
git lfs pull
```

---

## 📖 Usage

### Running the ManTacAi Suite (React + FastAPI)
The system now uses a modern **Client-Server Architecture**:
1.  **Backend**: FastAPI Server (Python) for heavy AI processing.
2.  **Frontend**: Next.js Dashboard (React) for real-time visualization.

#### 1. Start the Backend
```bash
cd backend
python main.py
```
*Server starts at `http://localhost:8000`*

#### 2. Start the Frontend
Open a new terminal:
```bash
cd frontend
npm install  # First time only
npm run dev
```
*Dashboard accessible at `http://localhost:3000`*

### Features (UI V2):
1.  **Holographic Dashboard**: Real-time Risk Radar, Phase Tracking, and "Glassmorphism" design.
2.  **Live Stream Analysis**: Paste logs or type directly to see per-message risk scoring.
3.  **Forensic Export**: One-click generation of professional court-ready reports.

---

## 📈 Performance Metrics

### Class-Level Accuracy (Test Set)

| Class | Precision | Recall | F1-Score | Status |
|-------|-----------|--------|----------|--------|
| **Gaslighting** | 99% | 99% | **0.99** | 🌟 Excellent |
| **Emergency** | 97% | 100% | **0.98** | 🛡️ Critical Safety |
| **Coercive Control** | 100% | 100% | **1.00** | 🤖 Likely Overfit |
| **Love Bombing** | 90% | 98% | **0.94** | ✅ Highly Sensitive |
| **Threats** | 91% | 68% | **0.78** | ⚠️ Needs Improvement |
| **Neutral** | 100% | 100% | **1.00** | ✅ No False Alarms |

**Average Inference Time:** 0.04s (Real-time capable)

---

## 📁 File Structure

```
ManTacAi/
├── app.py                            # Main Gradio Application Entry Point
├── requirements.txt                  # Python Dependencies
├── manipulation_tactic_detector_model/ # Fine-tuned V8 Model Weights (DistilRoBERTa)
│   ├── pytorch_model.bin
│   └── config.json
├── manipulation_detection/           # Core Logic Package
│   ├── src/
│   │   ├── inference/
│   │   │   ├── model.py             # Inference Pipeline
│   │   │   └── scoring.py           # Risk Scoring Logic
│   │   └── utils/
│   │       ├── context_engine.py    # Cycle of Abuse State Machine
│   │       ├── report.py            # Word Doc Generator
│   │       └── safety.py            # Emergency Keywords
├── scripts/                          # Utility & Training Scripts
└── README.md                         # This Documentation
```

---

## ⚖️ Disclaimer

**ManTacAi is a forensic analysis tool, not a clinical diagnostic instrument.**
It looks for *patterns* in text that match known manipulation tactics. It cannot diagnose Narcissistic Personality Disorder (NPD) or determine legal culpability.
*Always consult with a licensed mental health professional or legal counsel for serious situations.*

---

<p align="center">
  Made with ❤️ for Truth & Safety
</p>