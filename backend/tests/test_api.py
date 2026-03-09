"""
API integration tests for ManTacAi backend.
Tests cover the /api/analyze, /api/full-analysis, /api/reset, and / endpoints.
"""
import sys
import os
import pytest

# Path setup (mirrors main.py)
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)
sys.path.insert(0, os.path.join(backend_dir, "manipulation_detection", "src"))

from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


# --- HEALTH CHECK ---

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "ManTacAi" in data["system"]


# --- /api/analyze ---

class TestAnalyzeEndpoint:
    """Tests for the /api/analyze endpoint."""

    def test_empty_text_returns_safe(self):
        response = client.post("/api/analyze", json={"text": ""})
        assert response.status_code == 200
        data = response.json()
        assert data["risk_score"] == 0.0
        assert data["risk_level"] == "Safe"
        assert data["segments"] == []

    def test_simple_message(self):
        response = client.post("/api/analyze", json={
            "text": "Hello, how are you today?"
        })
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "risk_level" in data
        assert "segments" in data
        assert isinstance(data["segments"], list)
        assert len(data["segments"]) > 0

    def test_chat_log_format(self):
        chat = "Alex: You're crazy, that never happened.\nYou: I saw the messages.\nAlex: You're imagining things."
        response = client.post("/api/analyze", json={
            "text": chat,
            "suspect_name": "Alex"
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["segments"]) >= 2
        # Check that suspect was identified
        suspects = [s for s in data["segments"] if s["sender"] == "suspect"]
        assert len(suspects) > 0

    def test_context_factors(self):
        response = client.post("/api/analyze", json={
            "text": "You: I need some space.\nAlex: You're not going anywhere.",
            "suspect_name": "Alex",
            "context_factors": ["history_of_violence"]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["risk_score"] >= 0.0

    def test_stateless_mode(self):
        """Stateless requests should not persist state."""
        response = client.post("/api/analyze", json={
            "text": "Alex: I'm going to find you.",
            "suspect_name": "Alex",
            "stateless": True
        })
        assert response.status_code == 200

    def test_response_has_required_fields(self):
        response = client.post("/api/analyze", json={
            "text": "Person: Hello there"
        })
        assert response.status_code == 200
        data = response.json()
        required_fields = [
            "segments", "risk_score", "risk_level",
            "primary_pattern", "cycle_phase", "darvo_score",
            "timeline", "radar_chart_data"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_segment_structure(self):
        response = client.post("/api/analyze", json={
            "text": "Alex: You're being dramatic"
        })
        data = response.json()
        if data["segments"]:
            seg = data["segments"][0]
            assert "msg" in seg
            assert "sender" in seg
            assert "sender_name" in seg
            assert "risk_score" in seg
            assert "label" in seg
            assert "tactic_scores" in seg

    def test_text_too_long_rejected(self):
        """Text exceeding 50,000 chars should be rejected."""
        long_text = "a" * 50001
        response = client.post("/api/analyze", json={"text": long_text})
        assert response.status_code == 422  # Validation error

    def test_radar_chart_data_structure(self):
        response = client.post("/api/analyze", json={
            "text": "Alex: Do what I say or else."
        })
        data = response.json()
        for item in data["radar_chart_data"]:
            assert "subject" in item
            assert "A" in item
            assert "fullMark" in item
            assert item["fullMark"] == 100


# --- /api/full-analysis ---

class TestFullAnalysisEndpoint:
    """Tests for the /api/full-analysis narrative endpoint."""

    def test_safe_input(self):
        payload = {
            "risk_level": "Safe",
            "risk_score": 0.1,
            "darvo_score": 0.0,
            "primary_pattern": "None",
            "radar_chart_data": []
        }
        response = client.post("/api/full-analysis", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "narrative" in data
        assert isinstance(data["narrative"], str)

    def test_high_risk_input(self):
        payload = {
            "risk_level": "Critical",
            "risk_score": 0.95,
            "darvo_score": 0.8,
            "primary_pattern": "GASLIGHTING",
            "radar_chart_data": [
                {"subject": "Gaslighting", "A": 90, "fullMark": 100},
                {"subject": "Threats", "A": 70, "fullMark": 100}
            ]
        }
        response = client.post("/api/full-analysis", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["narrative"]) > 0
        assert "CRITICAL" in data["narrative"] or "manipulation" in data["narrative"].lower()


# --- /api/reset ---

class TestResetEndpoint:
    """Tests for the /api/reset endpoint."""

    def test_reset_returns_success(self):
        response = client.post("/api/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset"
