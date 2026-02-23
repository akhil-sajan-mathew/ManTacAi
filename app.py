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

# --- CUSTOM CSS: "Next Gen" Cyber-Noir Theme ---
NEXT_GEN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

/* Global Reset */
body, .gradio-container {
    background-color: #050510 !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
    background-image: radial-gradient(circle at 50% 10%, #1a1a40 0%, #050510 60%);
}

/* --- GLASSMORPHISM UTILITIES --- */
.glass-panel {
    background: rgba(20, 25, 40, 0.6) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(100, 200, 255, 0.08);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    border-radius: 16px;
    padding: 20px;
    transition: all 0.3s ease;
}

.glass-panel:hover {
    border-color: rgba(100, 200, 255, 0.2);
    box-shadow: 0 8px 32px 0 rgba(99, 102, 241, 0.15);
}

/* --- HOLOGRAPHIC HUD CARDS --- */
.hud-card {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
    border-top: 2px solid;
    position: relative;
    overflow: hidden;
}

.hud-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(to bottom, rgba(255,255,255,0.05) 0%, transparent 50%);
    pointer-events: none;
}

.hud-stat-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.5em;
    font-weight: 700;
    text-shadow: 0 0 10px currentColor;
    margin: 5px 0;
}

.hud-stat-label {
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 2px;
    opacity: 0.7;
}

/* --- CHAT VISUALIZATION --- */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 15px;
    max-height: 600px;
    overflow-y: auto;
    padding: 25px;
    background: rgba(10, 10, 20, 0.4); /* Darker container */
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}

.chat-bubble {
    padding: 16px 20px;
    border-radius: 12px;
    max-width: 85%;
    position: relative;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    animation: slideIn 0.3s ease-out;
    line-height: 1.5;
    text-align: left !important; /* Force readable text */
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Named: Suspect (Left) - Red Tint */
.bubble-left {
    align-self: flex-start;
    border-top-left-radius: 4px;
    background: linear-gradient(135deg, rgba(30, 0, 10, 0.8), rgba(20, 20, 30, 0.9));
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #ffd4d4;
}

/* Named: Victim (Right) - Blue Tint */
.bubble-right {
    align-self: flex-end;
    border-top-right-radius: 4px;
    background: linear-gradient(135deg, rgba(0, 10, 40, 0.8), rgba(20, 20, 30, 0.9));
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #dbeafe;
}

/* Anonymous: Intelligent Center - Neutral Grey */
.bubble-center {
    align-self: center;
    width: 95%; /* Make it wide like a log entry, but distinct */
    background: rgba(30, 41, 59, 0.7); /* Opaque enough to see */
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #f1f5f9;
}

/* High Risk Overrides */
.toxic-glow {
    background: linear-gradient(90deg, rgba(80, 0, 20, 0.6), rgba(30, 0, 0, 0.8)) !important;
    border: 1px solid #f43f5e !important;
    box-shadow: 0 0 12px rgba(244, 63, 94, 0.3) !important;
}

.risk-badge {
    font-size: 0.75em;
    font-weight: 800;
    letter-spacing: 1px;
    padding: 4px 8px;
    border-radius: 4px;
    margin-bottom: 8px;
    display: inline-block;
}

/* --- SIDEBAR & CONTROLS --- */
.sidebar-glass {
    background: rgba(10, 10, 20, 0.8) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

.glow-btn {
    background: linear-gradient(90deg, #ec4899 0%, #8b5cf6 100%) !important;
    border: none;
    box-shadow: 0 0 20px rgba(236, 72, 153, 0.4);
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 1px;
    transition: all 0.3s;
}
.glow-btn:hover {
    transform: scale(1.05);
    box-shadow: 0 0 30px rgba(236, 72, 153, 0.6);
}
"""

# --- INITIALIZATION ---
context_engine = ContextEngine()
try:
    model = ManipulationModel(model_path="manipulation_tactic_detector_model")
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"⚠️ Model Load Error: {e}")
    model = None

# --- CORE LOGIC ---

def parse_full_chat_log(log_text, suspect_name=""):
    """Parses chat log. Includes Robust Regex Splitter for blob inputs."""
    print(f"DEBUG: Parsing input of length {len(log_text)}...")
    
    # 0. Pre-processing: Force split on timestamps if newlines are missing
    # Look for [HH:MM] patterns that are NOT preceded by a newline
    # This adds a newline before every timestamp to ensure split works
    log_text = re.sub(r'([^\n])(\[?\d{1,2}:\d{2})', r'\1\n\2', log_text)
    
    # 1. Standard Split
    lines = log_text.split('\n')
    parsed_events = []
    current_ts = time.time()
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Timestamp Extraction
        match = re.search(r'\[?(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\]?', line)
        if match:
            try:
                dt_str = match.group(1)
                fmt = "%H:%M" if "M" not in dt_str.upper() else "%I:%M %p"
                dt = datetime.strptime(dt_str, fmt)
                current_ts = datetime.combine(date.today(), dt.time()).timestamp()
                line = re.sub(r'\[?(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\]?[:\-]?\s*', '', line).strip()
            except: pass
            
        # Filtering
        if suspect_name:
            # Check if line STARTS with suspect name (plus optional colon/hyphen)
            # Use regex to avoid matching "Project Morgan" inside a sentence
            if re.match(f"^{re.escape(suspect_name)}[:\\-]?\\s*", line, re.IGNORECASE):
                 clean_content = re.sub(f"^{re.escape(suspect_name)}[:\\-]?\\s*", "", line, flags=re.IGNORECASE)
                 if len(clean_content) >= 3:
                     parsed_events.append({'msg': clean_content, 'ts': current_ts, 'sender': 'suspect', 'raw': line})
            else:
                # Assume other lines are "Me"
                if len(line) >= 2:
                    parsed_events.append({'msg': line, 'ts': current_ts, 'sender': 'victim', 'raw': line})
        else:
            # Anonymous Mode
            if len(line) >= 2:
                parsed_events.append({'msg': line, 'ts': current_ts, 'sender': 'unknown', 'raw': line})
    
    print(f"DEBUG: Found {len(parsed_events)} valid events.")
    return parsed_events

def run_forensic_analysis(chat_log, safety_checklist, suspect_name=""):
    print("DEBUG: Analysis started...")
    events = parse_full_chat_log(chat_log, suspect_name)
    
    # Empty State Handling
    if not events:
        empty_fig = go.Figure()
        empty_fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return (empty_fig, empty_fig, "", "<div style='color:orange'>⚠️ No valid data.</div>", {}, gr.update(visible=False))

    # Analysis Loop
    aggregated_preds = {}
    history_risk = []
    running_state = "Neutral"
    
    # HTML Builder for Chat
    chat_html_accum = "<div class='chat-container'>"

    if model:
        for event in events:
            msg = event['msg']
            ts = event['ts']
            
            # Predict
            preds = model.predict(msg)
            detected_label = max(preds, key=preds.get)
            risk = preds[detected_label]
            
            # Aggregate
            for k, v in preds.items():
                aggregated_preds[k] = max(aggregated_preds.get(k, 0), v)
                
            # Context
            context_engine.add_event(msg, detected_label, risk, timestamp=ts)
            
            # History
            risk_score, _, _ = calculate_risk_score(preds)
            history_risk.append({"time": datetime.fromtimestamp(ts).strftime("%H:%M"), "risk": risk_score, "msg": msg[:30]})

            # --- BUILD CHAT BUBBLE HTML ---
            bubble_class = "bubble-center"
            
            # Determine alignment
            if event['sender'] == 'suspect':
                bubble_class = "bubble-left"
            elif event['sender'] == 'victim':
                bubble_class = "bubble-right"
            else: 
                # Smart Center Alignment logic
                # If high risk -> Left-ish? No, keep center but color it.
                bubble_class = "bubble-center"
            
            # Determine Glow/Color
            glow_class = ""
            badge_html = ""
            
            if risk > 0.65 and detected_label not in ['neutral_logistics', 'safe_general']:
                glow_class = "toxic-glow"
                badge_html = f"<div class='risk-badge' style='background: #f43f5e; color: white;'>⚠️ {detected_label.replace('_',' ').upper()} ({int(risk*100)}%)</div>"
            elif detected_label in ['safe_support', 'neutral_logistics']:
                glow_class = "safe-glow"
                
            chat_html_accum += f"""
            <div class='chat-bubble {bubble_class} {glow_class}'>
                {badge_html}
                <div class='msg-text'>{msg}</div>
            </div>
            """
            
    chat_html_accum += "</div>"
    
    # Final Metrics
    final_risk, _, final_pattern = calculate_risk_score(aggregated_preds)
    darvo = calculate_darvo_score(aggregated_preds)
    
    # FIX: Access state directly instead of missing summary() method
    running_state = context_engine.detector.state
    
    # Safety Overrides
    safety_risk, _, recs = evaluate_safety_risk(safety_checklist)
    if safety_risk in ["High", "Critical"]:
        final_risk = max(final_risk, 0.95)
        final_pattern = "⚠️ SAFETY ALARM"
        
    risk_level = "Critical" if final_risk > 0.85 else "High" if final_risk > 0.65 else "Moderate" if final_risk > 0.35 else "Safe"
    risk_color = "#f43f5e" if final_risk > 0.6 else "#fbbf24" if final_risk > 0.35 else "#10b981"

    # --- VISUALIZATION WRAPPING ---
    
    # 1. Gauge
    gauge_fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = final_risk * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 0},
            'bar': {'color': risk_color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [{'range': [0, 100], 'color': 'rgba(255,255,255,0.05)'}],
        },
        number = {'font': {'family': 'Orbitron', 'color': 'white'}}
    ))
    gauge_fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font={'color': "white"})

    # 2. Timeline
    df_hist = pd.DataFrame(history_risk)
    timeline_fig = go.Figure()
    if not df_hist.empty:
        timeline_fig.add_trace(go.Scatter(
            x=df_hist['time'], y=df_hist['risk'],
            mode='lines',
            line=dict(color='#a855f7', width=4, shape='spline'), # Neon Purple
            fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.1)'
        ))
    timeline_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color='rgba(255,255,255,0.5)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='white')
    )

    # 3. Holographic HUD HTML
    hud_html = f"""
    <div style="display: flex; gap: 20px; width: 100%; margin-bottom: 20px;">
        <div class="glass-panel hud-card" style="flex: 1; border-color: {risk_color};">
            <div class="hud-stat-label" style="color: {risk_color}">Risk Level</div>
            <div class="hud-stat-value" style="color: {risk_color}">{risk_level}</div>
        </div>
        <div class="glass-panel hud-card" style="flex: 1; border-color: #a855f7;">
            <div class="hud-stat-label" style="color: #a855f7">Current Phase</div>
            <div class="hud-stat-value" style="color: #a855f7">{running_state}</div>
        </div>
        <div class="glass-panel hud-card" style="flex: 1; border-color: #06b6d4;">
            <div class="hud-stat-label" style="color: #06b6d4">DARVO Index</div>
            <div class="hud-stat-value" style="color: #06b6d4">{darvo:.2f}</div>
        </div>
    </div>
    """

    return (
        gauge_fig, timeline_fig, hud_html, chat_html_accum,{}, gr.update(visible=True)
    )

# --- UI LAYOUT ---
theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate").set(
    body_background_fill="#050510",
    block_background_fill="#0f111a",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.1)"
)

demo = gr.Blocks()
demo.title = "ManTacAi Pro"
demo.theme = theme
demo.css = NEXT_GEN_CSS

with demo:
    state = gr.State({})
    
    with gr.Row():
        # Sidebar
        with gr.Column(scale=1, elem_classes="sidebar-glass"):
            gr.Markdown("## 🕵️ ManTacAi `PRO`")
            gr.Markdown("*Advanced Forensic Engine*")
            
            gr.Markdown("### 📡 Target Link")
            suspect_filter = gr.Textbox(label="Suspect Name", placeholder="e.g. John")
            
            gr.Markdown("### 🛡️ Parameters")
            chk_safety = gr.CheckboxGroup(choices=SAFETY_CHECKLIST_ITEMS, label="Safety Protocol")
            
            analyze_btn = gr.Button("INITIALIZE SCAN", elem_classes="glow-btn", size="lg")
            dl_btn = gr.Button("📄 Export Report", visible=False)
            
        # Main Stage
        with gr.Column(scale=4):
            # HUD Row
            stats_display = gr.HTML(visible=True)
            
            with gr.Tabs():
                with gr.TabItem("🧠 Neural Analysis"):
                    with gr.Row():
                        # Left: Chat Viz, Right: Graphs
                        with gr.Column(scale=1):
                            chat_visualizer = gr.HTML(label="Forensic Chat Stream", elem_classes="glass-panel")
                        with gr.Column(scale=1):
                            timeline_plot = gr.Plot(label="Volatility", elem_classes="glass-panel")
                            gauge_plot = gr.Plot(label="Threat Probability", elem_classes="glass-panel")
                    
                with gr.TabItem("💾 Data Input"):
                    chat_input = gr.Textbox(
                        label="Raw Log Data", 
                        placeholder="Paste export data here...",
                        lines=10, 
                        elem_classes="glass-panel"
                    )

    # Actions
    analyze_btn.click(
        run_forensic_analysis,
        inputs=[chat_input, chk_safety, suspect_filter],
        outputs=[gauge_plot, timeline_plot, stats_display, chat_visualizer, state, dl_btn]
    )

if __name__ == "__main__":
    demo.launch()
