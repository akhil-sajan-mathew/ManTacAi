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

ManTacAi uses a **Hybrid 3-Layer Logic** system to balance raw AI power with human-defined safety rules.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ManTacAi Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ Input Text   │───▶│ Preprocessor │───▶│ Layer 1: The Brain       │   │
│  │ (Chat Logs)  │    │              │    │ (Deep Learing Model)     │   │
│  └──────────────┘    │ • Normalize  │    │                          │   │
│                      │ • Tokenize   │    │ • DistilRoBERTa V8       │   │
│                      │ • Filter     │    │ • 18-Class Output        │   │
│                      │   (Len > 4)  │    │ • Probability Scores     │   │
│                      └──────────────┘    │                          │   │
│                                          └──────────┬───────────────┘   │
│                                                     │                   │
│                                                     ▼                   │
│  ┌──────────────┐                       ┌──────────────────────┐        │
│  │ Layer 3:     │◀──────────────────────│ Layer 2: Context     │        │
│  │ Safety Lock  │                       │ Engine (The Memory)  │        │
│  │ (Guardrails) │                       │                      │        │
│  └──────┬───────┘                       │ • Tracks "Cycle"     │        │
│         │                               │ • Adjusts Risk       │        │
│         │                               │   Thresholds         │        │
│         │                               └──────────┬───────────┘        │
│         │                                          │                    │
│         ▼                                          ▼                    │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │                     Forensic Output                          │       │
│  │  • Verdict: "High Risk - Gaslighting Pattern Detected"       │       │
│  │  • Evidence: "You're imagining things" (Confidence: 99%)     │       │
│  │  • Cycle State: "Explosion Phase"                            │       │
│  │  • Report: generated_report.docx                             │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 AI Models

### V8 Manipulation Detector (Fine-Tuned Transformer)

**Purpose**: Classify specific psychological tactics in conversational text.

**Architecture**: DistilRoBERTa-Base (Fine-tuned)

```
Input Layer         : Tokenized Text (Max Len 512)
├── Transformer Block 1 - 6 : Self-Attention Heads (12)
├── Dropout(0.1)
├── Classification Head
│   ├── Dense(768)
│   ├── Tanh Activation
│   ├── Dropout(0.1)
│   └── Dense(18) → [Gaslighting, Love Bombing, ..., Neutral]
└── Output: Softmax Probability Distribution
```

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **96.80%** |
| **Test F1 Score** | **96.71%** |
| Max Sequence | 512 Tokens |
| Inference Time | ~45ms per message (CPU) |

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

### Risk Assessment Logic
The raw probability is not enough. We calculate a weighted **Risk Score**:

```python
Risk Score = (Max_Prob * Severity_Weight)

# Severity Weights:
# - Urgent Emergency: 0.0 (Handled by Override)
# - Coercive Control: 1.0 (Critical)
# - Gaslighting:      0.9 (High)
# - Passive Aggress:  0.4 (Moderate)
```

### Context Engine (Cycle of Abuse)
The system maintains a rolling state window to detect the **Cycle of Abuse**:
1.  **Tension Building**: Rising frequency of Passive Aggression/Stonewalling.
2.  **Explosion**: High confidence Threats or Belligerence.
3.  **Honeymoon**: Sudden shift to Love Bombing/Apologies after an Explosion.

*If "Honeymoon" is detected within 10 messages of "Explosion", the Risk Score is forcibly elevated regardless of the message content.*

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

# 3. (Optional) Install LFS for Model Weights
git lfs install
git lfs pull
```

---

## 📖 Usage

### Running the Desktop App (GUI)
The primary interface is a local Gradio web app.

```bash
python app.py
```
*Creates a local server at `http://127.0.0.1:7860`*

### Features:
1.  **Paste & Analyze**: Copy complex chat logs into the text box.
2.  **Report Generation**: Click "Export Report" to get a `.docx` summary.
3.  **Pattern View**: See the visual distribution of tactics (e.g., "30% Gaslighting").

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