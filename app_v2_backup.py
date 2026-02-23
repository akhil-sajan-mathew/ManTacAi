import gradio as gr
import os
import sys
import re
import time
from datetime import datetime, date
import plotly.graph_objects as go
import pandas as pd

# Suppress tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path
sys.path.append(os.path.abspath(os.path.join("manipulation_detection", "src")))

from inference.model import ManipulationModel
from inference.scoring import calculate_risk_score, calculate_darvo_score
from utils.safety import evaluate_safety_risk, SAFETY_CHECKLIST_ITEMS, get_dynamic_safety_plan
from utils.report import generate_full_report
from utils.export import generate_word_report
from utils.context_engine import ContextEngine

# --- CUSTOM CSS: "Midnight Glass" Theme ---
MIDNIGHT_CSS = """
/* Background & Global Font */
body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

/* Glassmorphism Sidebar */
.sidebar-glass {
    background: rgba(17, 24, 39, 0.7) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Glass Panels */
.glass-panel {
    background: rgba(30, 41, 59, 0.4) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

/* Glowing Button */
.glow-btn {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    color: white !important;
    border: none;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.5);
    transition: all 0.3s ease;
    font-weight: bold;
    font-size: 1.1em;
}
.glow-btn:hover {
    box-shadow: 0 0 25px rgba(168, 85, 247, 0.7);
    transform: scale(1.02);
}

/* Inputs */
textarea, input {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f8fafc !important;
}

/* Typography Overrides */
h1, h2, h3 { color: white !important; }
.stat-value { font-size: 2em; font-weight: bold; }
.stat-label { font-size: 0.9em; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px; }

"""

# --- INITIALIZATION ---
context_engine = ContextEngine()
try:
    model = ManipulationModel(model_path="manipulation_tactic_detector_model")
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"⚠️ Model Load Error: {e}")
    model = None

# --- CORE LOGIC (Refactored) ---

def parse_full_chat_log(log_text, suspect_name=""):
    """Parses a raw chat log into a structured list of messages."""
    print(f"DEBUG: Parsing input of length {len(log_text)}...")
    lines = log_text.split('\n')
    parsed_events = []
    current_ts = time.time()
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 1. Timestamp Extraction
        match = re.search(r'\[?(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\]?', line)
        if match:
            try:
                dt_str = match.group(1)
                fmt = "%H:%M" if "M" not in dt_str.upper() else "%I:%M %p"
                dt = datetime.strptime(dt_str, fmt)
                current_ts = datetime.combine(date.today(), dt.time()).timestamp()
                # Clean prompt
                line = re.sub(r'\[?(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\]?[:\-]?\s*', '', line).strip()
            except: pass
            
        # 2. Sender Filtering (if name provided)
        if suspect_name:
            if suspect_name.lower() in line.lower().split(':')[0]: # Simple heuristic
                 # Try to remove name prefix
                 clean_content = re.sub(f"^{suspect_name}[:\\-]?\\s*", "", line, flags=re.IGNORECASE)
                 if len(clean_content) >= 3:
                     parsed_events.append((clean_content, current_ts))
        else:
            # LENIENT MODE: Accept any content > 2 chars
            if len(line) >= 2:
                parsed_events.append((line, current_ts))
    
    print(f"DEBUG: Found {len(parsed_events)} valid events.")
    return parsed_events

def run_forensic_analysis(chat_log, safety_checklist, suspect_name=""):
    print("DEBUG: Analysis started...")
    # 1. Parse
    events = parse_full_chat_log(chat_log, suspect_name)
    
    # Return empty fixtures if no data
    if not events:
        print("DEBUG: No events found. returning warnings.")
        empty_fig = go.Figure()
        empty_fig.update_layout(
             xaxis={"visible": False}, yaxis={"visible": False}, 
             annotations=[{"text": "No Data Found", "showarrow": False, "font": {"size": 20, "color": "gray"}}]
        )
        return (
            empty_fig, empty_fig, 
            "<div style='color:orange'>⚠️ No valid messages found. Please paste more text.</div>", 
            "**Status:** Waiting for input...", {}, gr.update(visible=False)
        )

    # 2. Analyze (Model + Context)
    aggregated_preds = {}
    history_risk = []
    running_state = "Neutral"
    
    print("DEBUG: Running model inference...")
    if model:
        for msg, ts in events:
            # Prediction
            preds = model.predict(msg)
            # Patch: Work Venting
            max_lbl = max(preds, key=preds.get)
            if max_lbl in ["belittling_ridicule", "threatening_intimidation"] and any(w in msg.lower() for w in ["boss", "game", "lag", "server"]):
                 if "you" not in msg.lower(): # Basic check
                     preds = {k:0.0 for k in preds} # Nullify
                     preds["benign_venting"] = 0.95
            
            # Aggregate
            for k, v in preds.items():
                aggregated_preds[k] = max(aggregated_preds.get(k, 0), v)
                
            # Context Update
            ctx_res = context_engine.add_event(msg, max(preds, key=preds.get), preds[max(preds, key=preds.get)], timestamp=ts)
            running_state = ctx_res["current_state"]
            
            # History for Graph
            risk_snapshot, _, _ = calculate_risk_score(preds)
            history_risk.append({"time": datetime.fromtimestamp(ts).strftime("%H:%M"), "risk": risk_snapshot, "msg": msg[:50]+"..."})
            
    # 3. Final Metrics
    final_risk, _, final_pattern = calculate_risk_score(aggregated_preds)
    darvo = calculate_darvo_score(aggregated_preds)
    
    # 4. Safety Overrides
    safety_risk, modifier, recs = evaluate_safety_risk(safety_checklist)
    if safety_risk in ["High", "Critical"]:
        final_risk = 0.95
        final_pattern = "⚠️ SAFETY: Coercive Control"
    elif final_pattern == "urgent_emergency":
        final_risk = 1.0
        final_pattern = "⚠️ EMERGENCY DETECTED"
        
    risk_level = "Critical" if final_risk > 0.85 else "High" if final_risk > 0.65 else "Moderate" if final_risk > 0.35 else "Safe"
    
    # --- VISUALIZATION GENERATION ---
    
    # A. Gauge Chart (Risk Score)
    gauge_fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = final_risk * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Probability"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#f43f5e" if final_risk > 0.6 else "#6366f1"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [65, 100], 'color': 'rgba(244, 63, 94, 0.3)'}
            ],
        }
    ))
    gauge_fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font={'color': "white", 'family': "Inter"})

    # B. Timeline Chart
    df_hist = pd.DataFrame(history_risk)
    timeline_fig = go.Figure()
    if not df_hist.empty:
        timeline_fig.add_trace(go.Scatter(
            x=df_hist['time'], 
            y=df_hist['risk'],
            mode='lines+markers',
            name='Risk Level',
            line=dict(color='#6366f1', width=3, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.2)',
            text=df_hist['msg']
        ))
        timeline_fig.update_layout(
            title="Emotional Volatility Timeline",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Inter"},
            xaxis=dict(showgrid=False, color='white'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
        )
    
    # C. Stats Cards (HTML)
    stats_html = f"""
    <div style="display: flex; gap: 20px; width: 100%;">
        <div class="glass-panel" style="flex: 1; text-align: center; border-left: 4px solid { '#f43f5e' if final_risk > 0.6 else '#10b981' };">
            <div class="stat-label">Verdict</div>
            <div class="stat-value" style="color: { '#f43f5e' if final_risk > 0.6 else '#10b981' }">{risk_level}</div>
        </div>
        <div class="glass-panel" style="flex: 1; text-align: center; border-left: 4px solid #a855f7;">
            <div class="stat-label">Cycle Phase</div>
            <div class="stat-value" style="color: #a855f7">{running_state}</div>
        </div>
        <div class="glass-panel" style="flex: 1; text-align: center; border-left: 4px solid #06b6d4;">
            <div class="stat-label">DARVO Score</div>
            <div class="stat-value" style="color: #06b6d4">{darvo:.2f}</div>
        </div>
    </div>
    """
    
    # State Mapping
    state_export = {
        "risk_level": risk_level, "pattern": final_pattern, "darvo_score": darvo, 
        "messages": [e[0] for e in events], "predictions": aggregated_preds, "safety_checklist": safety_checklist
    }
    
    return (
        gauge_fig, 
        timeline_fig, 
        stats_html, 
        f"### Primary Pattern Detected: {final_pattern}\n\n{recs[0] if recs else ''}",
        state_export,
        gr.update(visible=True)
    )

def export_doc(state):
    if not state: return None
    return generate_word_report(state)

# --- UI LAYOUT ---
theme = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="#0b0f19",
    block_background_fill="#1e293b",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.1)"
)

demo = gr.Blocks()
demo.title = "ManTacAi 2.0"
demo.theme = theme
demo.css = MIDNIGHT_CSS

with demo:
    state = gr.State({})
    
    with gr.Row():
        # === SIDEBAR ===
        with gr.Column(scale=1, elem_classes="sidebar-glass"):
            gr.Markdown("## 🕵️ ManTacAi v2.0")
            gr.Markdown("*Forensic Pattern Detection*")
            
            gr.Markdown("### ⚙️ Controls")
            suspect_filter = gr.Textbox(label="Suspect Name", placeholder="Filter by sender (e.g. John)")
            
            gr.Markdown("### 🛡️ Safety Checklist")
            chk_safety = gr.CheckboxGroup(choices=SAFETY_CHECKLIST_ITEMS, label="", elem_classes="checkbox-group")
            
            analyze_btn = gr.Button("🔍 ANALYZE LOGS", elem_classes="glow-btn", size="lg")
            
            gr.Markdown("---")
            dl_btn = gr.Button("📄 Download Report", visible=False)
            dl_file = gr.File(visible=False, label="Forensic Report")

        # === MAIN DASHBOARD ===
        with gr.Column(scale=4):
            # TABS
            with gr.Tabs():
                with gr.TabItem("📊 Dashboard"):
                    # Input Area
                    chat_input = gr.Textbox(
                        label="Conversation Log", 
                        placeholder="Paste chat history here (WhatsApp, iMessage export)...",
                        lines=5, 
                        elem_classes="glass-panel"
                    )
                    
                    # Top Stats Row
                    stats_display = gr.HTML(visible=True)
                    
                    # Visuals Row
                    with gr.Row():
                        timeline_plot = gr.Plot(label="Cycle Timeline", elem_classes="glass-panel")
                        gauge_plot = gr.Plot(label="Risk Gauge", elem_classes="glass-panel")
                    
                    # Report Summary
                    report_summary = gr.Markdown(elem_classes="glass-panel")
    
    # Actions
    analyze_btn.click(
        run_forensic_analysis,
        inputs=[chat_input, chk_safety, suspect_filter],
        outputs=[gauge_plot, timeline_plot, stats_display, report_summary, state, dl_btn]
    )
    
    dl_btn.click(
        export_doc,
        inputs=[state],
        outputs=[dl_file]
    )

if __name__ == "__main__":
    demo.launch()
