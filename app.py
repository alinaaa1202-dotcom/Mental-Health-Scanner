import re
from typing import List

try:
    from fastapi import FastAPI, HTTPException  # type: ignore[reportMissingImports]
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError("Missing required dependency 'fastapi'. Install with `pip install fastapi`") from exc

try:
    from pydantic import BaseModel, Field  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError("Missing required dependency 'pydantic'. Install with `pip install pydantic`") from exc

try:
    from transformers import pipeline  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError("Missing required dependency 'transformers'. Install with `pip install transformers`") from exc

app = FastAPI(title="MindGuard AI Core Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentiment_pipeline = None

@app.on_event("startup")
def load_model():
    global sentiment_pipeline
    sentiment_pipeline = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        return_all_scores=True
    )

CRISIS_KEYWORDS = [
    "suicide", "end my life", "self-harm", "kill myself", 
    "don't want to live", "want to die", "hurt myself"
]

class PassiveScanRequest(BaseModel):
    text: str = Field(..., min_length=3)
    permissions: dict = Field(default={"messages": True, "documents": True})

class MCQAnswer(BaseModel):
    question_id: int
    score: int = Field(..., ge=0, le=3)
    category: str

class AssessmentRequest(BaseModel):
    answers: List[MCQAnswer]

def extract_linguistic_markers(text: str) -> dict:
    words = re.findall(r'\b\w+\b', text.lower())
    total = len(words) if len(words) > 0 else 1
    
    first_person = len(re.findall(r'\b(i|me|my|myself|mine)\b', text.lower()))
    negations = len(re.findall(r'\b(no|not|never|neither|cannot|cant|wont|nothing)\b', text.lower()))
    absolutist = len(re.findall(r'\b(always|never|completely|impossible|every|everyone|nothing)\b', text.lower()))

    return {
        "first_person_ratio": round(first_person / total, 3),
        "negation_ratio": round(negations / total, 3),
        "absolutist_ratio": round(absolutist / total, 3)
    }

@app.post("/api/scan/passive")
async def analyze_passive_scan(payload: PassiveScanRequest):
    text = payload.text
    
    if any(kw in text.lower() for kw in CRISIS_KEYWORDS):
        return {
            "status": "CRISIS_TRIGGERED",
            "crisis_flag": True,
            "risk_level": "High",
            "primary_signal": "Immediate Crisis Indicators Detected",
            "message": "Critical distress markers found. Please connect with emergency hotline resources.",
            "emergency_contacts": {
                "national_line": "988",
                "text_line": "HOME to 741741"
            }
        }

    linguistics = extract_linguistic_markers(text)
    predictions = sentiment_pipeline(text)[0]
    
    neg_score = next((p['score'] for p in predictions if p['label'] == 'NEGATIVE'), 0.0)
    
    composite_score = (
        (neg_score * 0.5) +
        (linguistics["first_person_ratio"] * 0.25) +
        (linguistics["negation_ratio"] * 0.25)
    )
    composite_score = round(min(max(composite_score, 0.0), 1.0), 2)

    if composite_score >= 0.65:
        risk_level = "Elevated"
        primary_signal = "Elevated Stress & Anxiety Signals"
    elif composite_score >= 0.40:
        risk_level = "Moderate"
        primary_signal = "Mild Tension Detected"
    else:
        risk_level = "Low"
        primary_signal = "Normal Wellbeing Signals"

    return {
        "status": "SUCCESS",
        "crisis_flag": False,
        "mode": "Passive Scan",
        "primary_signal": primary_signal,
        "risk_level": risk_level,
        "metrics": {
            "stress": int(composite_score * 100),
            "anxiety": int((neg_score * 0.7 + linguistics["absolutist_ratio"] * 0.3) * 100),
            "depression_valence": int((1.0 - neg_score) * 100)
        },
        "linguistic_markers": linguistics,
        "privacy_note": "Raw input text processed in memory and immediately discarded."
    }

@app.post("/api/scan/assessment")
async def process_assessment(payload: AssessmentRequest):
    answers = payload.answers
    if not answers:
        raise HTTPException(status_code=400, detail="No answers provided.")

    stress_score = 0
    anxiety_score = 0
    depression_score = 0

    for ans in answers:
        weight = ans.score * 25
        if "stress" in ans.category:
            stress_score += weight
        if "anxiety" in ans.category:
            anxiety_score += weight
        if "depression" in ans.category:
            depression_score += weight

    stress_score = min(stress_score, 100)
    anxiety_score = min(anxiety_score, 100)
    depression_score = min(depression_score, 100)

    max_score = max(stress_score, anxiety_score, depression_score)
    
    if max_score < 35:
        risk_level = "Low"
        primary_signal = "Normal Wellbeing"
        recommendation = "Your answers indicate balanced baseline stress levels."
    elif max_score < 65:
        risk_level = "Moderate"
        recommendation = "Mild indicators of tension were detected. A short break or breathing exercise is recommended."
        if stress_score == max_score:
            primary_signal = "Elevated Stress"
        elif anxiety_score == max_score:
            primary_signal = "Anxiety Indicators Detected"
        else:
            primary_signal = "Low Mood / Depressive Signals"
    else:
        risk_level = "Elevated"
        recommendation = "Consistent high-stress/anxiety markers identified. Consider sharing these insights with a professional."
        if stress_score == max_score:
            primary_signal = "High Stress Level"
        elif anxiety_score == max_score:
            primary_signal = "Significant Anxiety Indicators"
        else:
            primary_signal = "Depressive Symptoms Detected"

    return {
        "status": "SUCCESS",
        "crisis_flag": False,
        "mode": "Rapid-Fire Assessment",
        "primary_signal": primary_signal,
        "risk_level": risk_level,
        "metrics": {
            "stress": stress_score,
            "anxiety": anxiety_score,
            "depression": depression_score
        },
        "recommendation": recommendation,
        "disclaimer": "This tool provides non-diagnostic screening markers, not a formal medical diagnosis."
    }