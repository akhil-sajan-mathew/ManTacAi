import sys
import os
import re
import time
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- PATH SETUP ---
# Ensure we can import from manipulation_detection
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "manipulation_detection", "src"))

from inference.model import ManipulationModel
from inference.scoring import calculate_risk_score, calculate_darvo_score
from inference.semantic_engine import SemanticAnalyzer
from utils.context_engine import ContextEngine

# --- INITIALIZATION ---
app = FastAPI(title="ManTacAi API", version="2.0.0")

# Enable CORS for React Frontend (Port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
try:
    # Model path relative to backend/
    model_path = os.path.join(base_dir, "manipulation_tactic_detector_model")
    detector_model = ManipulationModel(model_path=model_path)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"⚠️ Warning: Custom model not found, using default. Error: {e}")
    detector_model = ManipulationModel() # Fallback

# Initialize Semantic Engine (Phase 23)
semantic_analyzer = SemanticAnalyzer()
semantic_analyzer.compute_centroids(detector_model)

# Initialize Context Engine
context_path = os.path.join(base_dir, "context_state.json")
context_engine = ContextEngine(persistence_file=context_path)

# --- DATA MODELS ---

class AnalyzeRequest(BaseModel):
    text: str
    suspect_name: Optional[str] = ""
    stateless: bool = True # Default to True for testing stability
    context_factors: List[str] = [] # Context modifiers (e.g. "financial_dependency")

class ChatSegment(BaseModel):
    msg: str
    ts: float
    sender: str
    sender_name: str # The display name (e.g. "Alex_M")
    risk_score: float
    label: str
    timestamp_str: str
    tactic_scores: Dict[str, float] = {}
    darvo_score: float = 0.0

class AnalysisResponse(BaseModel):
    segments: List[ChatSegment]
    risk_score: float
    risk_level: str
    primary_pattern: str
    cycle_phase: str
    darvo_score: float
    timeline: List[Dict[str, Any]]
    radar_chart_data: List[Dict[str, Any]]

# --- UTILITIES ---

def parse_chat_log(log_text: str, suspect_name: str = "") -> List[Dict]:
    """
    Parses chat log using the robust regex logic
    """
    # 0. Blob Pre-processing
    log_text = re.sub(r'([^\n])(\[?\d{1,2}:\d{2})', r'\1\n\2', log_text)
    
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
            
        # Filtering / Attribution
        sender = "unknown"
        sender_name = "Subject B" # Default Display Name
        clean_content = line
        
        # Check for explicitly named sender (e.g. "Alex:" or "You:")
        name_match = re.match(r'^([^:\-]{1,20})[:\-]\s+', line)
        if name_match:
            extracted_name = name_match.group(1).strip()
            # Clip the name part
            clean_content = line[name_match.end():].strip()
            sender_name = extracted_name
            
            if suspect_name and suspect_name.lower() in extracted_name.lower():
                sender = "suspect"
            elif extracted_name.lower() in ["you", "me", "myself"]:
                 sender = "victim"
                 sender_name = "You"
            else:
                 # THIRD PARTY / OTHER PARTICIPANT
                 sender = "other"
        
        # Legacy Fallback (No name prefix, usually victim/you continuation or unformatted)
        elif suspect_name and re.match(f"^{re.escape(suspect_name)}", line, re.IGNORECASE):
             sender = "suspect"
             sender_name = suspect_name
        elif len(line) >= 2:
             # Default to "You" if it's just text without a name
             sender = "victim"
             sender_name = "You"

        if len(clean_content) >= 2:
            parsed_events.append({
                'msg': clean_content, 
                'ts': current_ts, 
                'sender': sender,
                'sender_name': sender_name
            })
            
    return parsed_events

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "system": "ManTacAi Forensic Engine"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_chat(request: AnalyzeRequest):
    events = parse_chat_log(request.text, request.suspect_name)
    
    if not events:
        # Return empty safe state
        return AnalysisResponse(
            segments=[],
            risk_score=0.0,
            risk_level="Safe",
            primary_pattern="None",
            cycle_phase="Normal",
            darvo_score=0.0,
            timeline=[]
        )

    processed_segments = []
    aggregated_preds = {}
    history_risk = []
    # Radar Chart Aggregation Buckets
    radar_metrics = {
        "Gaslighting": 0.0,
        "Guilt": 0.0,
        "Threats": 0.0,
        "Silence": 0.0,
        "Love Bomb": 0.0,
        "Deflection": 0.0
    }

    # Context Engine Selection
    if request.stateless:
        # Fresh instance for this request (Stateless Test Mode)
        active_engine = ContextEngine(persistence_file=None) 
    else:
        # Global instance (Legacy Surveillance Mode)
        active_engine = context_engine

    for event in events:
        msg = event['msg']
        
        # Predict
        preds, embedding = detector_model.predict(msg, return_embedding=True)
        detected_label = max(preds, key=preds.get)
        risk = preds[detected_label]
        
        # Phase 23: Semantic Check
        sem_score, sem_concept = semantic_analyzer.check_similarity(embedding)
        
        # Max-Score Aggregation for Radar
        radar_metrics["Gaslighting"] = max(radar_metrics["Gaslighting"], preds.get("gaslighting", 0))
        radar_metrics["Guilt"] = max(radar_metrics["Guilt"], preds.get("guilt_tripping", 0))
        radar_metrics["Threats"] = max(radar_metrics["Threats"], preds.get("threatening_intimidation", 0))
        radar_metrics["Silence"] = max(radar_metrics["Silence"], preds.get("stonewalling", 0))
        radar_metrics["Love Bomb"] = max(radar_metrics["Love Bomb"], preds.get("love_bombing", 0))
        radar_metrics["Deflection"] = max(radar_metrics["Deflection"], preds.get("deflection", 0))
        
        # Aggregate
        for k, v in preds.items():
            aggregated_preds[k] = max(aggregated_preds.get(k, 0), v)
            
        # Update Context Logic
        active_engine.add_event(msg, detected_label, risk, timestamp=event['ts'])
        
        # Calculate Segment Score
        seg_risk_score, _, _, seg_tactic_scores = calculate_risk_score(
            preds, 
            request.context_factors, 
            text_content=msg,
            semantic_data=(sem_score, sem_concept)
        )
        
        # Calculate Segment DARVO Contribution
        seg_darvo = calculate_darvo_score(seg_tactic_scores, msg)
        
        processed_segments.append(ChatSegment(
            msg=msg,
            ts=event['ts'],
            sender=event['sender'],
            sender_name=event.get('sender_name', 'Subject'),
            risk_score=seg_risk_score, 
            label=detected_label,
            timestamp_str=datetime.fromtimestamp(event['ts']).strftime("%H:%M"),
            tactic_scores=seg_tactic_scores,
            darvo_score=seg_darvo
        ))
        
        history_risk.append({
            "time": datetime.fromtimestamp(event['ts']).strftime("%H:%M"), 
            "risk": seg_risk_score
        })

    # Final Metrics
    final_text_blob = "\n".join([e['msg'] for e in events])
    final_risk, level, final_pattern, _ = calculate_risk_score(aggregated_preds, request.context_factors, text_content=final_text_blob)
    
    # Concatenate all text for DARVO analysis
    full_text_blob = "\n".join([e['msg'] for e in events])
    darvo = calculate_darvo_score(aggregated_preds, full_text_blob)
    
    running_state = active_engine.detector.state
    
    # Format Radar Data for Recharts
    formatted_radar = [
        {"subject": "Gaslighting", "A": int(radar_metrics["Gaslighting"] * 100), "fullMark": 100},
        {"subject": "Guilt", "A": int(radar_metrics["Guilt"] * 100), "fullMark": 100},
        {"subject": "Threats", "A": int(radar_metrics["Threats"] * 100), "fullMark": 100},
        {"subject": "Silence", "A": int(radar_metrics["Silence"] * 100), "fullMark": 100},
        {"subject": "Love Bomb", "A": int(radar_metrics["Love Bomb"] * 100), "fullMark": 100},
        {"subject": "Deflection", "A": int(radar_metrics["Deflection"] * 100), "fullMark": 100}
    ]
    
    return AnalysisResponse(
        segments=processed_segments,
        risk_score=final_risk,
        risk_level=level,
        primary_pattern=final_pattern,
        cycle_phase=running_state,
        darvo_score=darvo,
        timeline=history_risk,
        radar_chart_data=formatted_radar
    )

@app.post("/api/reset")
def reset_session():
    context_engine.reset()
    return {"status": "reset", "message": "Session memory cleared."}

if __name__ == "__main__":
    import uvicorn
    # Allow running directly for debug
    uvicorn.run(app, host="0.0.0.0", port=8000)
