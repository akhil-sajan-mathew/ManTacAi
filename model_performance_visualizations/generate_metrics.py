"""
ManTacAi End-to-End Pipeline Metrics Generator
Runs the full pipeline (ML + Lemmatization + Heuristics + Semantic Engine + Scoring)
on all test samples and generates 7 publication-quality PNG visualizations.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BACKEND_SRC = os.path.join(PROJECT_ROOT, 'backend', 'manipulation_detection', 'src')
sys.path.insert(0, BACKEND_SRC)

OUTPUT_DIR = SCRIPT_DIR  # Save PNGs alongside existing files

# --- IMPORTS FROM PROJECT ---
from inference.model import ManipulationModel
from inference.semantic_engine import SemanticAnalyzer
from inference.scoring import calculate_risk_score

# --- CONSTANTS ---
MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'manipulation_tactic_detector_model')
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset_augmented', 'v8_training_data_final.json')

ALL_LABELS = [
    "appeal_to_emotion", "belittling_ridicule", "benign_affection", "benign_venting",
    "coercive_control", "deflection", "ethical_persuasion", "gaslighting",
    "guilt_tripping", "healthy_conflict", "love_bombing", "neutral_conversation",
    "neutral_logistics", "passive_aggression", "stonewalling",
    "threatening_intimidation", "urgent_emergency", "whataboutism"
]

# Map pipeline override patterns back to standard tactic labels
OVERRIDE_MAP = {
    "SEMANTIC_PHYSICAL_VIOLENCE": "threatening_intimidation",
    "SEMANTIC_SELF_HARM": "threatening_intimidation",
    "GUILT TRIPPING": "guilt_tripping",
}

# --- STYLE CONFIG (Light Mode) ---
BG_COLOR = '#ffffff'
PANEL_COLOR = '#f1f5f9'
TEXT_COLOR = '#1e293b'
GRID_COLOR = '#cbd5e1'
ACCENT_INDIGO = '#4f46e5'
ACCENT_PURPLE = '#9333ea'
ACCENT_GREEN = '#059669'
ACCENT_YELLOW = '#ca8a04'
ACCENT_RED = '#dc2626'
ACCENT_CYAN = '#0891b2'
LABEL_GRAY = '#64748b'


def load_test_data():
    """Load test split from V8 dataset."""
    print("Loading test data...")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_samples = data['test']
    print(f"  Loaded {len(test_samples)} test samples.")
    return test_samples


def run_full_pipeline(model, semantic_analyzer, test_samples):
    """
    Run each test sample through the full pipeline and collect results.
    Returns a list of result dicts.
    """
    results = []
    total = len(test_samples)
    
    for i, sample in enumerate(test_samples):
        text = sample['text']
        true_label = sample['manipulation_tactic']
        
        # Step 1: ML Model + Embedding
        preds, embedding = model.predict(text, return_embedding=True)
        model_label = max(preds, key=preds.get)
        
        # Step 2: Semantic Check
        sem_score, sem_concept = semantic_analyzer.check_similarity(embedding)
        
        # Step 3: Full Scoring Pipeline (calls heuristics + semantic override internally)
        risk_score, risk_level, primary_pattern, refined_preds = calculate_risk_score(
            preds, text_content=text, semantic_data=(sem_score, sem_concept)
        )
        
        # Map pipeline output to standard label
        pipeline_label = OVERRIDE_MAP.get(primary_pattern, primary_pattern)
        
        # Determine what layer made the final decision
        if primary_pattern.startswith("SEMANTIC_"):
            override_source = "semantic"
        elif pipeline_label != model_label:
            override_source = "heuristic"
        else:
            override_source = "none"
        
        results.append({
            "true_label": true_label,
            "model_label": model_label,
            "pipeline_label": pipeline_label,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "override_source": override_source,
            "sem_score": sem_score,
        })
        
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  Processed {i+1}/{total} samples...")
    
    return results


# ===================== VISUALIZATION FUNCTIONS =====================

def render_metrics_table(results, output_path):
    """PNG 1: Styled performance metrics table."""
    true = [r['pipeline_label'] for r in results]
    pred = [r['pipeline_label'] for r in results]  # We compare true vs pipeline
    true_labels = [r['true_label'] for r in results]
    pred_labels = [r['pipeline_label'] for r in results]
    
    report = classification_report(true_labels, pred_labels, labels=ALL_LABELS,
                                   output_dict=True, zero_division=0)
    
    # Build table data
    rows = []
    for label in ALL_LABELS:
        m = report.get(label, {})
        rows.append([
            label.replace('_', ' ').title(),
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}",
            f"{m.get('f1-score', 0):.4f}",
            f"{int(m.get('support', 0))}"
        ])
    
    # Add summary rows
    rows.append(['', '', '', '', ''])  # Spacer
    acc = report.get('accuracy', 0)
    rows.append(['Accuracy', '', '', f'{acc:.4f}', f"{int(report['weighted avg']['support'])}"])
    macro = report.get('macro avg', {})
    rows.append(['Macro Avg', f"{macro.get('precision',0):.4f}", 
                 f"{macro.get('recall',0):.4f}", f"{macro.get('f1-score',0):.4f}",
                 f"{int(macro.get('support',0))}"])
    weighted = report.get('weighted avg', {})
    rows.append(['Weighted Avg', f"{weighted.get('precision',0):.4f}",
                 f"{weighted.get('recall',0):.4f}", f"{weighted.get('f1-score',0):.4f}",
                 f"{int(weighted.get('support',0))}"])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    
    col_labels = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    
    # Style cells
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        
        if row_idx == 0:  # Header
            cell.set_facecolor(ACCENT_INDIGO)
            cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        elif row_idx > len(ALL_LABELS):  # Summary rows
            cell.set_facecolor('#e0e7ff')
            cell.set_text_props(color=ACCENT_INDIGO, fontweight='bold')
        else:
            cell.set_facecolor(PANEL_COLOR)
            cell.set_text_props(color=TEXT_COLOR)
            
            # Color code F1 column (col 3)
            if col_idx == 3 and rows[row_idx - 1][3]:
                try:
                    f1_val = float(rows[row_idx - 1][3])
                    if f1_val >= 0.97:
                        cell.set_facecolor('#d1fae5')  # Light green
                    elif f1_val >= 0.93:
                        cell.set_facecolor('#fef3c7')  # Light yellow
                    elif f1_val > 0:
                        cell.set_facecolor('#fee2e2')  # Light red
                except ValueError:
                    pass
    
    ax.set_title('Table 1: End-to-End Pipeline Performance Metrics by Class',
                 color=TEXT_COLOR, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def render_confusion_matrix(results, output_path):
    """PNG 2: 18x18 confusion matrix heatmap."""
    true_labels = [r['true_label'] for r in results]
    pred_labels = [r['pipeline_label'] for r in results]
    
    cm = confusion_matrix(true_labels, pred_labels, labels=ALL_LABELS)
    
    fig, ax = plt.subplots(figsize=(18, 15))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Custom colormap
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    short_labels = [l.replace('_', '\n') for l in ALL_LABELS]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax, linewidths=0.5, linecolor=GRID_COLOR,
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', color=TEXT_COLOR, fontsize=12, labelpad=10)
    ax.set_ylabel('True Label', color=TEXT_COLOR, fontsize=12, labelpad=10)
    ax.set_title('Confusion Matrix: Full Pipeline (ML + Heuristics + Semantic)',
                 color=TEXT_COLOR, fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    
    # Color the colorbar text
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def render_f1_per_class(results, output_path):
    """PNG 3: Horizontal bar chart of F1-score per class."""
    true_labels = [r['true_label'] for r in results]
    pred_labels = [r['pipeline_label'] for r in results]
    
    report = classification_report(true_labels, pred_labels, labels=ALL_LABELS,
                                   output_dict=True, zero_division=0)
    
    f1_scores = {label: report.get(label, {}).get('f1-score', 0) for label in ALL_LABELS}
    sorted_labels = sorted(f1_scores.keys(), key=lambda x: f1_scores[x])
    sorted_scores = [f1_scores[l] for l in sorted_labels]
    
    # Color mapping
    colors = []
    for s in sorted_scores:
        if s >= 0.97:
            colors.append(ACCENT_GREEN)
        elif s >= 0.93:
            colors.append(ACCENT_YELLOW)
        else:
            colors.append(ACCENT_RED)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    display_labels = [l.replace('_', ' ').title() for l in sorted_labels]
    bars = ax.barh(display_labels, sorted_scores, color=colors, edgecolor=GRID_COLOR, height=0.6)
    
    # Add value labels
    for bar, score in zip(bars, sorted_scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', ha='left', color=TEXT_COLOR, fontsize=9)
    
    ax.set_xlim(0, 1.08)
    ax.set_xlabel('F1-Score', color=TEXT_COLOR, fontsize=11)
    ax.set_title('F1-Score per Class (Full Pipeline)', color=TEXT_COLOR, 
                 fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add threshold lines
    ax.axvline(x=0.97, color=ACCENT_GREEN, linestyle='--', alpha=0.5, label='0.97')
    ax.axvline(x=0.93, color=ACCENT_YELLOW, linestyle='--', alpha=0.5, label='0.93')
    ax.legend(loc='lower right', facecolor='white', edgecolor=GRID_COLOR, 
              labelcolor=TEXT_COLOR, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def render_pipeline_impact(results, output_path):
    """PNG 4: Stacked bar showing override source distribution."""
    df = pd.DataFrame(results)
    
    # Count by override source
    source_counts = df['override_source'].value_counts()
    ml_only = source_counts.get('none', 0)
    heuristic = source_counts.get('heuristic', 0)
    semantic = source_counts.get('semantic', 0)
    total = len(df)
    
    # Also compute accuracy per source
    df['correct'] = df['true_label'] == df['pipeline_label']
    ml_mask = df['override_source'] == 'none'
    heur_mask = df['override_source'] == 'heuristic'
    sem_mask = df['override_source'] == 'semantic'
    
    ml_acc = df[ml_mask]['correct'].mean() if ml_mask.sum() > 0 else 0
    heur_acc = df[heur_mask]['correct'].mean() if heur_mask.sum() > 0 else 0
    sem_acc = df[sem_mask]['correct'].mean() if sem_mask.sum() > 0 else 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    fig.patch.set_facecolor(BG_COLOR)
    
    # Left: Count pie chart
    ax1.set_facecolor(BG_COLOR)
    sizes = [ml_only, heuristic, semantic]
    pie_labels = [f'ML Only\n({ml_only})', f'Heuristic\nOverride ({heuristic})', f'Semantic\nOverride ({semantic})']
    pie_colors = [ACCENT_INDIGO, ACCENT_PURPLE, ACCENT_CYAN]
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=pie_labels, colors=pie_colors,
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'color': TEXT_COLOR, 'fontsize': 10},
                                         wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight('bold')
        at.set_color('white')
    ax1.set_title('Decision Source Distribution', color=TEXT_COLOR, fontsize=13, fontweight='bold')
    
    # Right: Accuracy per source
    ax2.set_facecolor(BG_COLOR)
    sources = ['ML Only', 'Heuristic', 'Semantic']
    accs = [ml_acc, heur_acc, sem_acc]
    counts = [ml_only, heuristic, semantic]
    bar_colors = [ACCENT_INDIGO, ACCENT_PURPLE, ACCENT_CYAN]
    
    bars = ax2.bar(sources, accs, color=bar_colors, edgecolor=GRID_COLOR, width=0.5)
    for bar, acc, cnt in zip(bars, accs, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{acc:.1%}\n(n={cnt})', ha='center', va='bottom', 
                 color=TEXT_COLOR, fontsize=10, fontweight='bold')
    
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Accuracy', color=TEXT_COLOR, fontsize=11)
    ax2.set_title('Accuracy by Decision Layer', color=TEXT_COLOR, fontsize=13, fontweight='bold')
    ax2.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax2.spines['bottom'].set_color(GRID_COLOR)
    ax2.spines['left'].set_color(GRID_COLOR)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def render_model_vs_pipeline(results, output_path):
    """PNG 5: Grouped bar chart comparing model-only vs pipeline accuracy per class."""
    df = pd.DataFrame(results)
    
    model_correct = {}
    pipeline_correct = {}
    
    for label in ALL_LABELS:
        mask = df['true_label'] == label
        n = mask.sum()
        if n > 0:
            model_correct[label] = (df[mask]['model_label'] == label).mean()
            pipeline_correct[label] = (df[mask]['pipeline_label'] == label).mean()
        else:
            model_correct[label] = 0
            pipeline_correct[label] = 0
    
    x = np.arange(len(ALL_LABELS))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    model_vals = [model_correct[l] for l in ALL_LABELS]
    pipe_vals = [pipeline_correct[l] for l in ALL_LABELS]
    
    bars1 = ax.bar(x - width/2, model_vals, width, label='Model Only', 
                   color=ACCENT_INDIGO, edgecolor=GRID_COLOR, alpha=0.8)
    bars2 = ax.bar(x + width/2, pipe_vals, width, label='Full Pipeline',
                   color=ACCENT_GREEN, edgecolor=GRID_COLOR, alpha=0.8)
    
    # Highlight improvements
    for i, (m, p) in enumerate(zip(model_vals, pipe_vals)):
        if p > m + 0.01:  # Pipeline improved
            ax.annotate(f'+{(p-m):.0%}', xy=(x[i] + width/2, p), xytext=(0, 5),
                       textcoords='offset points', ha='center', fontsize=7,
                       color=ACCENT_GREEN, fontweight='bold')
        elif m > p + 0.01:  # Pipeline regressed
            ax.annotate(f'{(p-m):.0%}', xy=(x[i] + width/2, p), xytext=(0, 5),
                       textcoords='offset points', ha='center', fontsize=7,
                       color=ACCENT_RED, fontweight='bold')
    
    ax.set_xticks(x)
    display_labels = [l.replace('_', '\n') for l in ALL_LABELS]
    ax.set_xticklabels(display_labels, fontsize=7, color=TEXT_COLOR, rotation=0, ha='center')
    ax.set_ylabel('Recall (per class)', color=TEXT_COLOR, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title('Model-Only vs Full Pipeline: Per-Class Recall',
                 color=TEXT_COLOR, fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='white', edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def render_risk_distribution(results, output_path):
    """PNG 6: Risk score distribution histogram."""
    df = pd.DataFrame(results)
    df['correct'] = df['true_label'] == df['pipeline_label']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    correct_scores = df[df['correct']]['risk_score']
    incorrect_scores = df[~df['correct']]['risk_score']
    
    bins = np.linspace(0, 1, 40)
    ax.hist(correct_scores, bins=bins, alpha=0.7, color=ACCENT_GREEN, 
            label=f'Correct ({len(correct_scores)})', edgecolor='white')
    ax.hist(incorrect_scores, bins=bins, alpha=0.7, color=ACCENT_RED,
            label=f'Incorrect ({len(incorrect_scores)})', edgecolor='white')
    
    # Threshold lines
    thresholds = [(0.3, 'Low/Med', '#6b7280'), (0.6, 'Med/High', ACCENT_YELLOW), (0.8, 'High/Crit', ACCENT_RED)]
    for thresh, name, color in thresholds:
        ax.axvline(x=thresh, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
        ax.text(thresh + 0.01, ax.get_ylim()[1] * 0.9, name, color=color, fontsize=8, rotation=90, va='top')
    
    ax.set_xlabel('Risk Score', color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel('Number of Samples', color=TEXT_COLOR, fontsize=11)
    ax.set_title('Risk Score Distribution (Correct vs Incorrect Predictions)',
                 color=TEXT_COLOR, fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='white', edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def render_overall_summary(results, output_path):
    """PNG 7: Overall summary card with headline numbers."""
    df = pd.DataFrame(results)
    
    true_labels = [r['true_label'] for r in results]
    pred_labels = [r['pipeline_label'] for r in results]
    model_labels = [r['model_label'] for r in results]
    
    report = classification_report(true_labels, pred_labels, labels=ALL_LABELS,
                                   output_dict=True, zero_division=0)
    model_report = classification_report(true_labels, model_labels, labels=ALL_LABELS,
                                          output_dict=True, zero_division=0)
    
    pipeline_acc = report['accuracy']
    model_acc = model_report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    n_total = len(results)
    n_heuristic = sum(1 for r in results if r['override_source'] == 'heuristic')
    n_semantic = sum(1 for r in results if r['override_source'] == 'semantic')
    acc_delta = pipeline_acc - model_acc
    
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'ManTacAi — End-to-End Pipeline Performance Summary',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=18, fontweight='bold', color=TEXT_COLOR)
    
    # Metrics grid (2 rows x 4 cols)
    metrics = [
        ('Pipeline\nAccuracy', f'{pipeline_acc:.2%}', ACCENT_GREEN),
        ('Model-Only\nAccuracy', f'{model_acc:.2%}', ACCENT_INDIGO),
        ('Pipeline\nDelta', f'{acc_delta:+.2%}', ACCENT_CYAN if acc_delta >= 0 else ACCENT_RED),
        ('Test\nSamples', f'{n_total}', LABEL_GRAY),
        ('Macro\nF1-Score', f'{macro_f1:.4f}', ACCENT_PURPLE),
        ('Weighted\nF1-Score', f'{weighted_f1:.4f}', ACCENT_PURPLE),
        ('Heuristic\nOverrides', f'{n_heuristic}', ACCENT_YELLOW),
        ('Semantic\nOverrides', f'{n_semantic}', ACCENT_CYAN),
    ]
    
    cols = 4
    for i, (label, value, color) in enumerate(metrics):
        row = i // cols
        col = i % cols
        x = 0.125 + col * 0.22
        y = 0.62 - row * 0.32
        
        # Card background
        rect = FancyBboxPatch((x - 0.08, y - 0.1), 0.18, 0.22,
                              boxstyle="round,pad=0.01", 
                              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
                              transform=ax.transAxes, linewidth=1)
        ax.add_patch(rect)
        
        # Value
        ax.text(x + 0.01, y + 0.06, value, transform=ax.transAxes,
                ha='center', va='center', fontsize=20, fontweight='bold', color=color)
        # Label
        ax.text(x + 0.01, y - 0.04, label, transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color=LABEL_GRAY, linespacing=1.3)
    
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ===================== MAIN =====================

def main():
    print("=" * 60)
    print("  ManTacAi End-to-End Pipeline Metrics Generator")
    print("=" * 60)
    
    # 1. Load model
    print("\n[1/3] Loading model...")
    model = ManipulationModel(model_path=MODEL_PATH)
    
    # 2. Initialize semantic engine
    print("\n[2/3] Initializing Semantic Engine...")
    semantic = SemanticAnalyzer()
    semantic.compute_centroids(model)
    
    # 3. Load test data
    test_samples = load_test_data()
    
    # 4. Run full pipeline
    print("\n[3/3] Running full pipeline on test set...")
    results = run_full_pipeline(model, semantic, test_samples)
    
    # 5. Generate all visualizations
    print("\n" + "=" * 60)
    print("  Generating Visualizations...")
    print("=" * 60)
    
    render_metrics_table(results, os.path.join(OUTPUT_DIR, 'performance_metrics_table.png'))
    render_confusion_matrix(results, os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    render_f1_per_class(results, os.path.join(OUTPUT_DIR, 'f1_per_class.png'))
    render_pipeline_impact(results, os.path.join(OUTPUT_DIR, 'pipeline_impact.png'))
    render_model_vs_pipeline(results, os.path.join(OUTPUT_DIR, 'model_vs_pipeline.png'))
    render_risk_distribution(results, os.path.join(OUTPUT_DIR, 'risk_distribution.png'))
    render_overall_summary(results, os.path.join(OUTPUT_DIR, 'overall_summary.png'))
    
    # 6. Save raw data as CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, 'detailed_metrics.csv')
    
    true_labels = [r['true_label'] for r in results]
    pred_labels = [r['pipeline_label'] for r in results]
    report_df = pd.DataFrame(
        classification_report(true_labels, pred_labels, labels=ALL_LABELS,
                              output_dict=True, zero_division=0)
    ).transpose()
    report_df.to_csv(csv_path)
    print(f"  Saved: {csv_path}")
    
    print("\n" + "=" * 60)
    print("  DONE! All 7 PNGs generated in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
