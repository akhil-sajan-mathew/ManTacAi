import sys
import os
import re
import time
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# --- PATH SETUP ---
# Add manipulation_detection/src to path for internal imports
# TODO: Migrate to proper pip install -e . once internal imports use relative style
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "manipulation_detection", "src"))

from inference.model import ManipulationModel
from inference.scoring import calculate_risk_score, calculate_darvo_score
from inference.context_scoring import compute_attribution
from inference.semantic_engine import SemanticAnalyzer
from utils.context_engine import ContextEngine
from utils.action_handlers.narrative_generator import generate_narrative_summary

# Dependency providers
from dependencies import get_detector_model, get_semantic_analyzer, get_context_engine

# --- INITIALIZATION ---
app = FastAPI(title="ManTacAi API", version="2.0.0")

# Enable CORS for React Frontend
allowed_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API KEY AUTHENTICATION (Optional) ---
# Set MANTACAI_API_KEY env var to enable; unset to disable (local dev)
_api_key = os.environ.get("MANTACAI_API_KEY")

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Skip auth if no API key is configured, or for health/CORS preflight
    if not _api_key or request.method == "OPTIONS" or request.url.path == "/":
        return await call_next(request)

    provided_key = request.headers.get("X-API-Key")
    if provided_key != _api_key:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

    return await call_next(request)

# --- RATE LIMITING ---
import collections

_rate_limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
_request_log: dict = collections.defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple in-memory rate limiter for the analyze endpoint."""
    if request.method != "POST" or "/api/analyze" not in request.url.path:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Clean old entries (older than 60s)
    _request_log[client_ip] = [t for t in _request_log[client_ip] if now - t < 60]

    if len(_request_log[client_ip]) >= _rate_limit:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded. Max {_rate_limit} requests per minute."}
        )

    _request_log[client_ip].append(now)
    return await call_next(request)


# Eagerly load models at startup to fail fast
@app.on_event("startup")
def startup_load_models():
    get_detector_model()
    get_semantic_analyzer()

# --- DATA MODELS ---

class AnalyzeRequest(BaseModel):
    text: str
    suspect_name: Optional[str] = ""
    stateless: bool = True # Default to True for testing stability
    context_factors: List[str] = [] # Context modifiers (e.g. "financial_dependency")

    @field_validator('text')
    @classmethod
    def text_must_not_be_too_long(cls, v):
        if len(v) > 50000:
            raise ValueError('Text input exceeds maximum length of 50,000 characters')
        return v

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
    role: str = "neutral"           # "initiator" | "reactor" | "neutral"
    initiated_risk: float = 0.0     # risk attributed to unprovoked manipulation
    reactive_risk: float = 0.0      # risk attributed to defensive echo

class AnalysisResponse(BaseModel):
    segments: List[ChatSegment]
    risk_score: float
    risk_level: str
    primary_pattern: str
    cycle_phase: str
    darvo_score: float
    timeline: List[Dict[str, Any]]
    radar_chart_data: List[Dict[str, Any]]
    speaker_attribution: Dict[str, Dict[str, Any]] = {}

# --- UTILITIES ---

def parse_chat_log(log_text: str, suspect_name: str = ""):
    lines = log_text.split('\n')
    parsed_events = []
    
    current_speaker = "Subject B" # Default
    base_ts = time.time()
    msg_counter = 0  # For incrementing timestamps when no real ts is available
    
    # Timestamp extraction patterns
    re_ts_wa = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2})\s*(AM|PM|am|pm)?')
    re_ts_bracket = re.compile(r'^\[(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?)\]')
    re_ts_bracket_time = re.compile(r'^\[(\d{1,2}:\d{2})\s*(AM|PM|am|pm)?\]')

    # 1. Bracketed: [anything] Name: Message
    re_bracket = re.compile(r'^\[.*?\]\s*([^:\-]+)[:\-]\s*(.*)$')
    # 2. Discord: Name — Date at Time
    re_discord = re.compile(r'^([^—\-]+)\s+[—\-]\s+(?:Today at|Yesterday at|[0-9/]+)\s+\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?$', re.IGNORECASE)
    # 3. Standard WA text: Date, Time - Name: Message
    re_wa = re.compile(r'^[\d/,:\s]+(?:AM|PM|am|pm)?\s*-\s*([^:]+):\s*(.*)$')
    # 4. Simple: Name: Message
    re_simple = re.compile(r'^([a-zA-Z0-9_ \-]{2,20})[:\-]\s+(.*)$')
    
    def _extract_timestamp(line):
        """Try to extract a Unix timestamp from a line. Returns float or None."""
        # WhatsApp: 12/5/23, 10:30 AM
        m = re_ts_wa.match(line)
        if m:
            date_str = m.group(1)
            time_str = m.group(2)
            ampm = m.group(3) or ""
            try:
                for fmt in ("%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p", 
                            "%m/%d/%y %H:%M", "%m/%d/%Y %H:%M",
                            "%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p",
                            "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M"):
                    try:
                        dt = datetime.strptime(f"{date_str} {time_str} {ampm}".strip(), fmt)
                        return dt.timestamp()
                    except ValueError:
                        continue
            except Exception:
                pass

        # Bracketed: [2024-01-15 10:30]
        m = re_ts_bracket.match(line)
        if m:
            try:
                dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
                return dt.timestamp()
            except ValueError:
                try:
                    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    return dt.timestamp()
                except ValueError:
                    pass

        # Bracketed time only: [10:30 AM]
        m = re_ts_bracket_time.match(line)
        if m:
            try:
                time_str = m.group(1)
                ampm = m.group(2) or ""
                fmt = "%I:%M %p" if ampm else "%H:%M"
                dt = datetime.strptime(f"{time_str} {ampm}".strip(), fmt)
                # Use today's date + parsed time
                today = date.today()
                dt = dt.replace(year=today.year, month=today.month, day=today.day)
                return dt.timestamp()
            except ValueError:
                pass

        return None

    for line in lines:
        line = line.strip()
        if not line: continue
        
        is_header = False
        msg_content = line
        speaker_match = None
        
        # Extract timestamp from the line
        parsed_ts = _extract_timestamp(line)
        if parsed_ts:
            current_ts = parsed_ts
        else:
            # Increment by 1 second per message to maintain ordering
            msg_counter += 1
            current_ts = base_ts + msg_counter
        
        m_disc = re_discord.match(line)
        if m_disc:
            speaker_match = m_disc.group(1).strip()
            msg_content = ""
            is_header = True
        else:
            m_brack = re_bracket.match(line)
            if m_brack:
                speaker_match = m_brack.group(1).strip()
                msg_content = m_brack.group(2).strip()
                is_header = True
            else:
                m_wa = re_wa.match(line)
                if m_wa:
                    speaker_match = m_wa.group(1).strip()
                    msg_content = m_wa.group(2).strip()
                    is_header = True
                else:
                    m_simple = re_simple.match(line)
                    if m_simple:
                        speaker_match = m_simple.group(1).strip()
                        msg_content = m_simple.group(2).strip()
                        is_header = True
        
        if is_header and speaker_match:
            current_speaker = speaker_match
            
        if not msg_content:
            continue
            
        sender = "unknown"
        lower_speaker = current_speaker.lower()
        if suspect_name and suspect_name.lower() in lower_speaker:
            sender = "suspect"
        elif lower_speaker in ["you", "me", "myself", "subject a"]:
            sender = "victim"
            current_speaker = "You"
        elif lower_speaker == "subject b":
            sender = "unknown" # Keep default logic
        else:
            sender = "other"
            
        if parsed_events and parsed_events[-1]['sender_name'] == current_speaker:
            parsed_events[-1]['msg'] += " " + msg_content
        else:
            parsed_events.append({
                'msg': msg_content,
                'sender': sender,
                'sender_name': current_speaker,
                'ts': current_ts
            })
            
    return parsed_events

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "system": "ManTacAi Forensic Engine"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_chat(
    request: AnalyzeRequest,
    detector_model: ManipulationModel = Depends(get_detector_model),
    semantic_analyzer: SemanticAnalyzer = Depends(get_semantic_analyzer),
    context_engine: ContextEngine = Depends(get_context_engine),
):
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
            timeline=[],
            radar_chart_data=[],
            speaker_attribution={}
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

    # Sliding window buffers for context-aware analysis
    window_size = 3
    window_embeddings = []   # CLS embeddings for echo detection
    window_senders = []      # sender names for echo detection
    window_risks = []        # calculated risk scores for attribution
    window_labels = []       # detected tactic labels for label-based aggression detection

    # Context Engine Selection
    if request.stateless:
        # Fresh instance for this request (Stateless Test Mode)
        active_engine = ContextEngine(persistence_file=None) 
    else:
        # Global instance (Legacy Surveillance Mode)
        active_engine = context_engine

    for idx, event in enumerate(events):
        msg = event['msg']
        sender_name = event.get('sender_name', 'Subject')
        
        # --- DUAL-PASS PREDICTION ---
        # Build context window (last N messages from the conversation)
        context_msgs = [e['msg'] for e in events[max(0, idx - window_size):idx]]
        
        isolated_preds, contextual_preds, embedding = detector_model.predict_with_context(
            msg, context_msgs, return_embedding=True
        )
        
        # Use isolated predictions for tactic scoring (what the text SAYS)
        preds = isolated_preds
        detected_label = max(preds, key=preds.get)
        risk = preds[detected_label]
        
        # Phase 23: Semantic Check
        sem_score, sem_concept = semantic_analyzer.check_similarity(embedding)
        
        # Calculate Segment Score FIRST (needed for accurate window_risks)
        seg_risk_score, _, _, seg_tactic_scores = calculate_risk_score(
            preds, 
            request.context_factors, 
            text_content=msg,
            semantic_data=(sem_score, sem_concept)
        )
        
        # --- ECHO DETECTION ---
        echo_sim, is_echo = semantic_analyzer.check_echo(
            embedding,
            window_embeddings[-window_size:],
            window_senders[-window_size:],
            sender_name
        )
        
        # --- IDENTIFY PRIMARY AGGRESSOR ---
        # Look at all speaker profiles to find the one with the highest demonstrated severity.
        # This prevents the system from locking onto a victim who is just reacting defensively.
        primary_aggressor = None
        max_severity = 0.0
        for name, profile in active_engine.speaker_profiles.items():
            # Severity requires at least one high-risk initiation to prevent false positives
            severity = profile.high_risk_count + profile.avg_risk
            if severity > max_severity and profile.high_risk_count >= 1:
                max_severity = severity
                primary_aggressor = name
                
        is_primary = (sender_name == primary_aggressor)
        
        # --- CONTEXT-AWARE ATTRIBUTION ---
        # Uses seg_risk_score (calculated risk) not raw model probability
        # Pass primary aggressor status + preceding labels for asymmetric dampening
        attribution = compute_attribution(
            isolated_preds=isolated_preds,
            contextual_preds=contextual_preds,
            preceding_risks=window_risks[-window_size:],
            preceding_senders=window_senders[-window_size:],
            preceding_labels=window_labels[-window_size:],
            current_sender=sender_name,
            is_echo=is_echo,
            echo_similarity=echo_sim,
            is_primary_aggressor=is_primary,
        )
        role = attribution["role"]
        dampening_factor = attribution["dampening_factor"]
        
        # Update sliding window buffers AFTER using them
        window_embeddings.append(embedding)
        window_senders.append(sender_name)
        window_risks.append(seg_risk_score)
        window_labels.append(detected_label)
        
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
            
        # Update Context Logic (use calculated risk, not raw probability)
        active_engine.add_event(msg, detected_label, seg_risk_score, timestamp=event['ts'])
        
        # Update Speaker Profile (use calculated risk for accurate tracking)
        active_engine.update_speaker_profile(sender_name, detected_label, seg_risk_score, role)
        
        # Apply context-aware dampening to reactor risk
        adjusted_risk = seg_risk_score * dampening_factor
        
        # Calculate Segment DARVO Contribution
        seg_darvo = calculate_darvo_score(seg_tactic_scores, msg)
        
        processed_segments.append(ChatSegment(
            msg=msg,
            ts=event['ts'],
            sender=event['sender'],
            sender_name=sender_name,
            risk_score=adjusted_risk, 
            label=detected_label,
            timestamp_str=datetime.fromtimestamp(event['ts']).strftime("%H:%M"),
            tactic_scores=seg_tactic_scores,
            darvo_score=seg_darvo,
            role=role,
            initiated_risk=attribution["initiated_risk"],
            reactive_risk=attribution["reactive_risk"]
        ))
        
        history_risk.append({
            "time": datetime.fromtimestamp(event['ts']).strftime("%H:%M"), 
            "risk": adjusted_risk
        })

    # Final Metrics
    final_text_blob = "\n".join([e['msg'] for e in events])
    final_risk, level, final_pattern, _ = calculate_risk_score(aggregated_preds, request.context_factors, text_content=final_text_blob)
    
    # Concatenate all text for DARVO analysis
    full_text_blob = "\n".join([e['msg'] for e in events])
    darvo = calculate_darvo_score(aggregated_preds, full_text_blob)
    
    running_state = active_engine.detector.state
    
    # Get Speaker Attribution
    speaker_attribution = active_engine.get_speaker_profiles()
    
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
        radar_chart_data=formatted_radar,
        speaker_attribution=speaker_attribution
    )

class NarrativeResponse(BaseModel):
    narrative: str

@app.post("/api/full-analysis", response_model=NarrativeResponse)
async def get_full_analysis(request: dict):
    """
    Consumes the metrics output payload from the frontend to generate 
    a contextual read-out of the abuse dynamics.
    """
    narrative = generate_narrative_summary(request)
    return NarrativeResponse(narrative=narrative)


@app.post("/api/reset")
def reset_session(context_engine: ContextEngine = Depends(get_context_engine)):
    context_engine.reset()
    return {"status": "reset", "message": "Session memory cleared."}

if __name__ == "__main__":
    import uvicorn
    # Allow running directly for debug
    uvicorn.run(app, host="0.0.0.0", port=8000)
