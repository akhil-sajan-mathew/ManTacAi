import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# 1. Setup
MODEL_PATH = "manipulation_tactic_detector_model"
DATA_PATH = "dataset_augmented/v8_training_data_final.json"
OUTPUT_DIR = "model_performance_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Hardcoded Historical Data (From User Image)
history = {
    "epoch": [1, 2, 3, 4],
    "val_loss": [0.644780, 0.392446, 0.311120, 0.286655],
    "accuracy": [0.810241, 0.882530, 0.913655, 0.917671],
    "f1": [0.784739, 0.884511, 0.913971, 0.918169]
}

def plot_training_history():
    print("Generating Training History Graphs...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["val_loss"], marker='o', linewidth=2, color='#e74c3c')
    plt.title("Model Convergence: Validation Loss", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xticks(history["epoch"])
    plt.savefig(f"{OUTPUT_DIR}/validation_loss_curve.png")
    plt.close()
    
    # Plot 2: Accuracy & F1
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["accuracy"], marker='o', label='Accuracy', linewidth=2, color='#2ecc71')
    plt.plot(history["epoch"], history["f1"], marker='s', label='F1 Score', linewidth=2, color='#3498db')
    plt.title("Model Improvement: Accuracy & F1 Score", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(history["epoch"])
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/accuracy_f1_curve.png")
    plt.close()

# 3. Fresh Evaluation
def run_evaluation():
    print("Loading Model & Data for Fresh Evaluation...")
    
    # Load Data
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Extract Texts and Labels
    # Note: Using 'train' key as that's where the data is
    df = pd.DataFrame(raw_data['train'])
    
    # Load Model to get label mappings
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    id2label = model.config.id2label
    label2id = model.config.label2id
    
    # Filter valid data
    df = df[df['manipulation_tactic'].isin(label2id.keys())]
    
    # Create Test Split (20%)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['manipulation_tactic'])
    
    print(f"Evaluating on {len(test_df)} examples...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    batch_size = 16
    texts = test_df['text'].tolist()
    labels = test_df['manipulation_tactic'].tolist()
    
    # Inference Loop
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
        y_pred.extend([id2label[p] for p in preds])
        y_true.extend(batch_labels)
        
    # Generate Confusion Matrix
    print("Plotting Confusion Matrix...")
    labels_list = sorted(list(label2id.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_list, yticklabels=labels_list)
    plt.title("Confusion Matrix: Where does the Model get confused?", fontsize=18)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    plt.close()
    
    # Generate Classification Report
    print("Generating Metrics Report...")
    report = classification_report(y_true, y_pred, target_names=labels_list, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{OUTPUT_DIR}/detailed_metrics.csv")
    
    # Plot F1 Score per Class
    f1_scores = report_df.loc[labels_list]['f1-score'].sort_values()
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x=f1_scores.values, y=f1_scores.index, palette='viridis')
    plt.title("F1-Score by Class (Model Reliability)", fontsize=16)
    plt.xlabel("F1 Score")
    plt.axvline(x=0.8, color='r', linestyle='--', alpha=0.5, label='High Reliability Threshold (0.8)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/f1_score_per_class.png")
    plt.close()
    
    # Generate Label Count Graph
    print("Plotting Label Distribution...")
    plt.figure(figsize=(12, 8))
    sns.countplot(y=labels, order=pd.Series(labels).value_counts().index, palette='pastel')
    plt.title("Test Set Label Distribution", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/label_distribution.png")
    plt.close()
    
    print(f"Done! All visualizations saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    plot_training_history()
    run_evaluation()
