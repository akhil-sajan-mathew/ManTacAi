import gradio as gr
import os
import sys
import re

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
from datetime import datetime, date
import time

# Initialize Context Engine (Persistent State)
context_engine = ContextEngine()

# Initialize Model (Lazy loading or global)
# We'll initialize it globally for this demo, but ideally it should be cached
try:
    # Try to load fine-tuned model if available, else base model
    model_path = "manipulation_tactic_detector_model"
    model = ManipulationModel(model_path=model_path)
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

def parse_timestamp(line):
    # Try to find [HH:MM] or HH:MM AM/PM
    # Returns float timestamp (epoch) or None
    match = re.search(r'\[?(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\]?', line)
    if match:
        time_str = match.group(1)
        try:
            dt = datetime.strptime(time_str, "%H:%M")
        except ValueError:
            try:
                dt = datetime.strptime(time_str, "%I:%M %p")
            except ValueError:
                return None
        
        # Combine with today's date
        full_dt = datetime.combine(date.today(), dt.time())
        return full_dt.timestamp()
    return None

def is_benign(text):
    # 1. Filter short garbage (< 3 words)
    if len(text.split()) < 3:
        return True
    
    # 2. Whitelist of safe greetings
    safe_terms = {"hello", "hi", "hey", "ok", "okay", "thanks", "thank you", "yes", "no"}
    cleaned = re.sub(r'[^\w\s]', '', text.lower()).strip()
    if cleaned in safe_terms:
        return True
        
    return False

def get_risk_verdict(score):
    if score < 0.35: return "Low Risk / Safe"
    if score < 0.65: return "Moderate Risk"
    if score < 0.85: return "High Risk"
    return "Critical Risk"

def analyze_messages(msg1, msg2, msg3, safety_checklist, suspect_name=""):
    """
    Main analysis function called by the UI.
    Supports Sender Parsing AND Timestamp Parsing.
    """
    # Combine inputs and split by lines
    all_text = "\n".join([m for m in [msg1, msg2, msg3] if m])
    lines = all_text.split('\n')
    
    # Pre-processing: Extract valid messages with metadata
    valid_events = [] # List of (text, timestamp)
    current_timestamp = time.time() # Default
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 1. Parse Timestamp
        ts = parse_timestamp(line)
        if ts:
            current_timestamp = ts
            # Strip timestamp for cleaner text analysis
            line = re.sub(r'\[?(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\]?[:\-]?\s*', '', line).strip()

        # 2. Sender Parsing
        if suspect_name:
            pattern = re.compile(f"^[\\[\\(]?{re.escape(suspect_name)}[\\]\\)]?[:\\-]?\\s*", re.IGNORECASE)
            if pattern.match(line):
                content = pattern.sub("", line).strip()
                if len(content) >= 4 and re.search(r'[a-zA-Z]', content):
                    valid_events.append((content, current_timestamp))
        else:
            if len(line) >= 4 and re.search(r'[a-zA-Z]', line):
                valid_events.append((line, current_timestamp))
                
    messages = [m for m, t in valid_events] # For compatibility with return


    if not messages:
        # If input was provided but filtered out (garbage)
        if all_text.strip():
            warning_msg = "⚠️ **Input Ignored**\n\nThe text provided is too short or doesn't look like a real message. Please enter meaningful sentences (minimum 4 characters) for analysis."
            return (
                gr.update(visible=False), 
                gr.update(value=warning_msg, visible=True), # Use Concerns box for warning
                gr.update(visible=False), 
                gr.update(visible=False), 
                gr.update(visible=False),
                {}
            )
        # No input at all
        return (
            gr.update(visible=False), # Risk Card
            gr.update(visible=False), # Key Concerns
            gr.update(visible=False), # Additional Analysis
            gr.update(visible=False), # Recommendations
            gr.update(visible=False),  # Timeline
            {}
        )

    # 1. Safety Check (Override)
    safety_risk_level, risk_modifier, safety_recs = evaluate_safety_risk(safety_checklist)
    is_high_risk = safety_risk_level in ["High", "Critical"]
    
    # 2. Sequential Analysis Loop
    final_risk_level = "Low Risk / Safe" # Default
    final_risk_score = 0.0
    final_pattern = "None"
    final_cycle_state = "Neutral" # Default for benign/safe
    final_darvo = 0.0
    
    aggregated_predictions_all = {} # For reporting

    if model:
        # We process sequentially to update the State Machine per message
        for msg_text, timestamp in valid_events:
            # BENIGN FILTER
            if is_benign(msg_text):
                continue # Skip model analysis for safe inputs

            # Predict
            preds = model.predict(msg_text)
            
            # --- CONTEXT PATCH: Detect Work/Gaming Venting ---
            # Prevents "I hate this game" or "My boss is an idiot" from being flagged as Ridicule/Threats
            temp_max_label = max(preds, key=preds.get)
            toxic_labels = ["belittling_ridicule", "threatening_intimidation", "passive_aggression"]
            
            if temp_max_label in toxic_labels:
                work_triggers = ["boss", "job", "work", "manager", "coworker", "client", "customer"]
                # Note: 'trash' excluded for safety.
                tech_triggers = ["game", "level", "dev", "developer", "lag", "glitch", "server", "computer", "wifi", "internet", "phone", "app"] 
                all_context_triggers = work_triggers + tech_triggers
                
                text_lower = msg_text.lower()
                
                # Check if anger is context-based (Safe)
                if any(t in text_lower for t in all_context_triggers):
                    # SAFETY CHECK: Ensure it's not blaming the partner ("Because of you")
                    if "because of you" not in text_lower:
                        # Override to Benign Venting (Safe)
                        # We reconstruct preds to ensure global aggregation sees this as SAFE.
                        preds = {k: 0.0 for k in preds} 
                        preds["benign_venting"] = 0.95 # High confidence safe
            # -------------------------------------------------

            # Aggregate for global report
            for label, prob in preds.items():
                aggregated_predictions_all[label] = max(aggregated_predictions_all.get(label, 0.0), prob)
            
            # Find primary label for this event
            max_label = max(preds, key=preds.get)
            max_score = preds[max_label]
            
            # Feed Context Engine
            context_result = context_engine.add_event(
                msg_text,
                max_label, 
                max_score,
                timestamp=timestamp
            )
            
            # Update running state
            final_cycle_state = context_result["current_state"]
            
        # Recalculate Global Metrics based on Aggregation + Final Context
        if aggregated_predictions_all:
             final_risk_score, _, final_pattern = calculate_risk_score(aggregated_predictions_all)
             final_darvo = calculate_darvo_score(aggregated_predictions_all)
             
             # Verbalize Verdict
             final_risk_level = get_risk_verdict(final_risk_score)

             # --- V8 EMERGENCY OVERRIDE ---
             # If the primary detected pattern is an Emergency (e.g. "Call 911"), 
             # force the UI to alert mode, even if "Manipulation Risk" is 0.0.
             if final_pattern == "urgent_emergency":
                 final_risk_level = "⚠️ EMERGENCY DETECTED"
                 final_risk_score = 1.0 # Force red color logic
                 # Note: We keep the pattern as "urgent_emergency"

             # Contextual Overrides (Only if significant risk detected AND not emergency)
             elif final_risk_score >= 0.55:
                if final_cycle_state == "EXPLOSION":
                    final_risk_level = "Critical Risk"
                    final_risk_score = max(final_risk_score, 0.95)
                    final_pattern = f"{final_pattern} (Explosion Phase)"
                elif final_cycle_state == "HONEYMOON":
                    final_risk_level = "High Risk" 
                    final_pattern = "Manipulation Cycle: Honeymoon Phase"
                elif final_cycle_state == "TENSION":
                    if "Low" in final_risk_level:
                        final_risk_level = "Moderate Risk"
                        final_pattern = "Tension Building"
             else:
                 # Force Neutral phase logic if score is low
                 final_cycle_state = "Neutral"
        else:
             # All inputs were benign
             final_risk_level = "Low Risk / Safe"
             final_risk_score = 0.0
             final_cycle_state = "Neutral"
                
    else:
        final_risk_score, final_risk_level, final_pattern = 0.0, "Unknown", "Model Error"
        final_darvo = 0.0
        
    # Map final variables to legacy names for UI construction
    risk_level = final_risk_level
    risk_score = final_risk_score
    detected_pattern = final_pattern
    darvo_score = final_darvo
    cycle_state = final_cycle_state

    # Safety Override / Modifier
    if is_high_risk:
        risk_level = safety_risk_level # Use the safety level (Critical/High)
        risk_score = max(risk_score, 0.95 if safety_risk_level == "Critical" else 0.8)
        detected_pattern = "Coercive Control / Safety Concern"
    else:
        # Apply modifier for lower level safety concerns if any
        risk_score = min(1.0, risk_score + risk_modifier)
        
    # Construct Output
    
    # 1. Risk Card
    # 1. Risk Card
    risk_color = "#ef4444" if "High" in risk_level or "Critical" in risk_level else "#eab308" if "Moderate" in risk_level else "#22c55e"
    risk_html = f"""
    <div style="background-color: {risk_color}20; border: 2px solid {risk_color}; border-radius: 10px; padding: 20px; text-align: center;">
        <h2 style="color: {risk_color}; margin: 0;">{risk_level}</h2>
        <p style="color: white; margin-top: 5px;">Based on the messages you shared</p>
        <div style="background-color: {risk_color}40; padding: 10px; border-radius: 5px; margin-top: 15px;">
            <h3 style="color: white; margin: 0;">{detected_pattern}</h3>
            <p style="color: white; margin-top: 5px; font-weight: bold;">Cycle Phase: {cycle_state}</p>
            <p style="color: white; margin: 0;">Risk Score: {int(risk_score * 100)}%</p>
        </div>
    </div>
    """

    # 2. Key Concerns
    concerns_text = f"**{detected_pattern}**\n\n"
    if is_high_risk and safety_recs:
        concerns_text += "⚠️ **SAFETY ALERT**:\n"
        for rec in safety_recs:
            concerns_text += f"* {rec}\n"
        concerns_text += "\n"
    concerns_text += "Analysis indicates potential manipulation tactics. "

    # 3. Additional Analysis (DARVO)
    darvo_level = "High" if darvo_score > 0.7 else "Moderate" if darvo_score > 0.4 else "Low"
    darvo_html = f"""
    <div style="background-color: #1f2937; color: white; padding: 15px; border-radius: 8px; border: 1px solid #374151;">
        <strong style="color: #fcd34d;">📊 DARVO Score: {darvo_score:.2f} ({darvo_level})</strong><br>
        <span style="font-size: 0.9em; opacity: 0.9;">DARVO (Deny, Attack, Reverse Victim & Offender) indicates potential narrative manipulation.</span>
    </div>
    """

    # 4. Recommendations
    recommendations = """
    *   Continue monitoring communication patterns that concern you.
    *   Consider discussing communication styles with your partner when you feel safe to do so.
    *   Trust your memory and perceptions - consider keeping notes.
    """
    if is_high_risk:
        recommendations = """
        *   **Prioritize your safety.** Consider contacting a support hotline.
        *   Do not confront the partner if you feel unsafe.
        *   Create a safety plan.
        """

    # 5. Timeline (Mock Plot)
    # import plotly.graph_objects as go
    # fig = go.Figure(...) 
    # For now, return None or a placeholder
    
    return (
        gr.update(value=risk_html, visible=True),
        gr.update(value=concerns_text, visible=True),
        gr.update(value=darvo_html, visible=True),
        gr.update(value=recommendations, visible=True),
        gr.update(visible=True), # Timeline placeholder
        {
            "risk_level": risk_level, 
            "pattern": detected_pattern, 
            "darvo_score": darvo_score,
            "messages": messages,
            "predictions": aggregated_predictions_all if model else {},
            "safety_checklist": safety_checklist
        } 
    )

def generate_safety_plan(analysis_state):
    if not analysis_state:
        return gr.update(visible=False, value="")
    
    plan = get_dynamic_safety_plan(
        analysis_state.get("risk_level", "Low"),
        analysis_state.get("pattern", "Unknown"),
        analysis_state.get("darvo_score", 0.0)
    )
    return gr.update(value=plan, visible=True)

def show_full_analysis(analysis_state):
    if not analysis_state:
        return gr.update(visible=False, value="")
        
    report = generate_full_report(
        analysis_state.get("messages", []),
        analysis_state.get("predictions", {}),
        analysis_state.get("risk_level", "Unknown"),
        analysis_state.get("pattern", "Unknown"),
        analysis_state.get("darvo_score", 0.0),
        analysis_state.get("safety_checklist", [])
    )
    return gr.update(value=report, visible=True)

def download_report(analysis_state):
    if not analysis_state:
        return None
    
    file_path = generate_word_report(analysis_state)
    return gr.update(value=file_path, visible=True)


# Custom CSS for Dark Theme
custom_css = """
body { background-color: #0b0f19; color: white; }
.gradio-container { background-color: #0b0f19 !important; }
h1, h2, h3, p, span { color: white !important; }
.input-box textarea { background-color: #1f2937 !important; color: white !important; border: 1px solid #374151 !important; }
.checkbox-group label { color: white !important; background-color: #1f2937 !important; }
.analyze-btn { background-color: #f97316 !important; color: white !important; border: none !important; }
.analyze-btn:hover { background-color: #ea580c !important; }
"""

with gr.Blocks() as demo:
    gr.HTML(f"<style>{custom_css}</style>")
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 2.5em; margin-bottom: 10px;">Manipulation Tactic Detector</h1>
        <p style="font-size: 1.1em; opacity: 0.8;">Share messages that concern you, and we'll help you understand what patterns might be present.</p>
    </div>
    """)

    # State for storing analysis results
    analysis_state = gr.State({})

    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=1):
            gr.Markdown("### Share Your Messages")
            gr.Markdown("Enter up to three messages that made you feel uncomfortable, confused, or concerned.")
            
            msg1 = gr.Textbox(label="Message 1 *", placeholder="e.g., \"You never take responsibility for your actions.\"", lines=3, elem_classes="input-box")
            msg2 = gr.Textbox(label="Message 2 (optional)", placeholder="Enter the message here...", lines=3, elem_classes="input-box")
            msg3 = gr.Textbox(label="Message 3 (optional)", placeholder="Enter the message here...", lines=3, elem_classes="input-box")
            
            suspect_name = gr.Textbox(label="Suspect Name (Optional)", placeholder="Enter name to filter chat logs (e.g. 'John')", lines=1, elem_classes="input-box")


        # Right Column: Safety Checklist
        with gr.Column(scale=1):
            gr.Markdown("### Safety Checklist")
            gr.Markdown("Optional but recommended. Check any that apply to your situation:")
            
            safety_checklist = gr.CheckboxGroup(
                choices=SAFETY_CHECKLIST_ITEMS,
                label="",
                elem_classes="checkbox-group"
            )

    # Analyze Button
    analyze_btn = gr.Button("Analyze Messages", size="lg", elem_classes="analyze-btn")

    # Output Section
    gr.Markdown("### Analysis Results", visible=True)
    
    with gr.Row():
        # Risk Card
        risk_output = gr.HTML(visible=False)
        
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Key Concerns Found", visible=False)
            concerns_output = gr.Markdown(visible=False)
            
            gr.Markdown("#### Additional Analysis", visible=False)
            darvo_output = gr.HTML(visible=False)

        with gr.Column():
            gr.Markdown("#### Personalized Recommendations", visible=False)
            recommendations_output = gr.Markdown(visible=False)
            with gr.Row():
                safety_btn = gr.Button("🛡️ Get Safety Plan")
                full_analysis_btn = gr.Button("📄 Show Full Analysis")
                download_btn = gr.Button("⬇️ Download Report")
            
            # Dynamic Safety Plan Output
            safety_plan_output = gr.Markdown(visible=False)
            
            # Full Analysis Output
            full_analysis_output = gr.Markdown(visible=False)
            
            # Download File Output (Hidden until generated)
            download_output = gr.File(label="Download Report", visible=False, file_types=[".docx"])

    # Timeline Graph (Placeholder)
    timeline_output = gr.Plot(visible=False, label="Pattern Timeline")

    # Wiring
    analyze_btn.click(
        analyze_messages,
        inputs=[msg1, msg2, msg3, safety_checklist, suspect_name],
        outputs=[risk_output, concerns_output, darvo_output, recommendations_output, timeline_output, analysis_state]
    )

    safety_btn.click(
        generate_safety_plan,
        inputs=[analysis_state],
        outputs=[safety_plan_output]
    )

    full_analysis_btn.click(
        show_full_analysis,
        inputs=[analysis_state],
        outputs=[full_analysis_output]
    )

    download_btn.click(
        download_report,
        inputs=[analysis_state],
        outputs=[download_output]
    )

if __name__ == "__main__":
    demo.launch()
