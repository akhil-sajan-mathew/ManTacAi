import gradio as gr
import os
import sys

# Suppress tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path
sys.path.append(os.path.abspath(os.path.join("manipulation_detection", "src")))

from inference.model import ManipulationModel
from inference.scoring import calculate_risk_score, calculate_darvo_score
from utils.safety import evaluate_safety_risk, SAFETY_CHECKLIST_ITEMS

# Initialize Model (Lazy loading or global)
# We'll initialize it globally for this demo, but ideally it should be cached
try:
    # Try to load fine-tuned model if available, else base model
    model_path = "manipulation_tactic_detector_model"
    model = ManipulationModel(model_path=model_path)
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

def analyze_messages(msg1, msg2, msg3, safety_checklist):
    """
    Main analysis function called by the UI.
    """
    messages = [m for m in [msg1, msg2, msg3] if m.strip()]
    
    if not messages:
        return (
            gr.update(visible=False), # Risk Card
            gr.update(visible=False), # Key Concerns
            gr.update(visible=False), # Additional Analysis
            gr.update(visible=False), # Recommendations
            gr.update(visible=False)  # Timeline
        )

    # 1. Safety Check (Override)
    safety_risk_level, risk_modifier, safety_recs = evaluate_safety_risk(safety_checklist)
    is_high_risk = safety_risk_level in ["High", "Critical"]
    
    # 2. Model Inference
    if model:
        # Predict for each message
        batch_results = model.predict_batch(messages)
        
        # Aggregate results (e.g., take max probability across messages for each label)
        aggregated_predictions = {}
        for res in batch_results:
            for label, prob in res.items():
                aggregated_predictions[label] = max(aggregated_predictions.get(label, 0.0), prob)
                
        # Calculate Scores
        risk_score, risk_level, detected_pattern = calculate_risk_score(aggregated_predictions)
        darvo_score = calculate_darvo_score(aggregated_predictions)
        
    else:
        # Fallback if model fails
        risk_score, risk_level, detected_pattern = 0.0, "Unknown", "Model Error"
        darvo_score = 0.0

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
    risk_color = "#ef4444" if risk_level in ["High", "Critical"] else "#eab308" if risk_level == "Medium" else "#22c55e"
    risk_html = f"""
    <div style="background-color: {risk_color}20; border: 2px solid {risk_color}; border-radius: 10px; padding: 20px; text-align: center;">
        <h2 style="color: {risk_color}; margin: 0;">{risk_level} Risk</h2>
        <p style="color: white; margin-top: 5px;">Based on the messages you shared</p>
        <div style="background-color: {risk_color}40; padding: 10px; border-radius: 5px; margin-top: 15px;">
            <h3 style="color: white; margin: 0;">{detected_pattern} pattern detected</h3>
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
    <div style="background-color: #fef08a; color: #854d0e; padding: 15px; border-radius: 8px;">
        <strong>📊 DARVO Score: {darvo_score:.2f} ({darvo_level})</strong><br>
        <span style="font-size: 0.9em;">DARVO (Deny, Attack, Reverse Victim & Offender) indicates potential narrative manipulation.</span>
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
        gr.update(visible=True) # Timeline placeholder
    )


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

    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=1):
            gr.Markdown("### Share Your Messages")
            gr.Markdown("Enter up to three messages that made you feel uncomfortable, confused, or concerned.")
            
            msg1 = gr.Textbox(label="Message 1 *", placeholder="e.g., \"You never take responsibility for your actions.\"", lines=3, elem_classes="input-box")
            msg2 = gr.Textbox(label="Message 2 (optional)", placeholder="Enter the message here...", lines=3, elem_classes="input-box")
            msg3 = gr.Textbox(label="Message 3 (optional)", placeholder="Enter the message here...", lines=3, elem_classes="input-box")

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
                gr.Button("🛡️ Get Safety Plan")
                gr.Button("📄 Show Full Analysis")
                gr.Button("⬇️ Download Report")

    # Timeline Graph (Placeholder)
    timeline_output = gr.Plot(visible=False, label="Pattern Timeline")

    # Wiring
    analyze_btn.click(
        analyze_messages,
        inputs=[msg1, msg2, msg3, safety_checklist],
        outputs=[risk_output, concerns_output, darvo_output, recommendations_output, timeline_output]
    )

if __name__ == "__main__":
    demo.launch()
