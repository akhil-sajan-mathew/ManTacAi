# Manipulation Detection Model

A transformer-based text classification model for detecting psychological manipulation tactics in text.

## Overview

This project implements a machine learning model that can identify 10 different manipulation tactics plus ethical persuasion in text:

- Gaslighting
- Guilt-tripping
- Deflection
- Stonewalling
- Belittling/Ridicule
- Love-bombing
- Threatening/Intimidation
- Passive-aggression
- Appeal to emotion
- Whataboutism
- Ethical persuasion

## Project Structure

```
manipulation_detection/
├── config/
│   └── model_config.yaml      # Model and training configuration
├── src/
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Model architecture
│   ├── training/              # Training pipeline
│   ├── evaluation/            # Evaluation and metrics
│   ├── inference/             # Inference and prediction
│   └── utils/                 # Utility functions
├── scripts/                   # Training and evaluation scripts
├── tests/                     # Unit and integration tests
├── models/                    # Saved model checkpoints
├── logs/                      # Training logs and metrics
└── requirements.txt           # Python dependencies
```

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have the dataset files in the `dataset/` directory

## Usage

### Training

```bash
python scripts/train.py
```

### Evaluation

```bash
python scripts/evaluate.py --model_path models/best_model.pt
```

### Inference

```bash
python scripts/predict.py --text "Your text here"
```

## Model Architecture

- Base Model: DistilBERT (distilbert-base-uncased)
- Classification Head: Linear layer with 11 outputs
- Input: Tokenized text (max 512 tokens)
- Output: Softmax probabilities for each manipulation tactic

## Performance Target

- Overall Accuracy: 85%+ on test set
- Balanced precision/recall across all classes
- Confidence scores for predictions
