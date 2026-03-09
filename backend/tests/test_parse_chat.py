"""
Unit tests for the parse_chat_log function.
Covers WhatsApp, Discord, bracketed, and simple chat formats.
"""
import sys
import os
import pytest

# Path setup
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)
sys.path.insert(0, os.path.join(backend_dir, "manipulation_detection", "src"))

from main import parse_chat_log


class TestSimpleFormat:
    """Tests for 'Name: Message' format."""

    def test_basic_parse(self):
        log = "Alex: Hello there\nYou: Hi Alex"
        result = parse_chat_log(log)
        assert len(result) == 2

    def test_sender_names(self):
        log = "Alex: Hello\nYou: Hi"
        result = parse_chat_log(log)
        assert result[0]["sender_name"] == "Alex"
        assert result[1]["sender_name"] == "You"

    def test_victim_identification(self):
        log = "You: I feel scared\nAlex: You're overreacting"
        result = parse_chat_log(log)
        assert result[0]["sender"] == "victim"

    def test_suspect_identification(self):
        log = "Alex: You're crazy\nYou: No I'm not"
        result = parse_chat_log(log, suspect_name="Alex")
        assert result[0]["sender"] == "suspect"

    def test_consecutive_messages_merged(self):
        log = "Alex: First part\nSecond part of the same message"
        result = parse_chat_log(log)
        # Both lines get attributed to Alex, merged into one
        assert len(result) == 1
        assert "First part" in result[0]["msg"]
        assert "Second part" in result[0]["msg"]


class TestWhatsAppFormat:
    """Tests for WhatsApp export format."""

    def test_wa_format(self):
        log = "12/5/23, 10:30 AM - Alex: Hey are you there\n12/5/23, 10:31 AM - You: Yes what's up"
        result = parse_chat_log(log)
        assert len(result) == 2
        assert result[0]["sender_name"] == "Alex"

    def test_wa_suspect(self):
        log = "1/15/24, 3:45 PM - Alex: Don't talk to them\n1/15/24, 3:46 PM - Me: Why not"
        result = parse_chat_log(log, suspect_name="Alex")
        suspects = [r for r in result if r["sender"] == "suspect"]
        assert len(suspects) >= 1


class TestBracketedFormat:
    """Tests for '[timestamp] Name: Message' format."""

    def test_bracketed_format(self):
        log = "[10:30 AM] Alex: You need to listen to me\n[10:31 AM] You: I am listening"
        result = parse_chat_log(log)
        assert len(result) == 2
        assert result[0]["sender_name"] == "Alex"

    def test_bracketed_with_date(self):
        log = "[2024-01-15 10:30] Alex: Where were you\n[2024-01-15 10:31] You: At work"
        result = parse_chat_log(log)
        assert len(result) == 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_input(self):
        result = parse_chat_log("")
        assert result == []

    def test_whitespace_only(self):
        result = parse_chat_log("   \n  \n  ")
        assert result == []

    def test_single_line(self):
        result = parse_chat_log("Alex: Hello")
        assert len(result) == 1

    def test_no_speaker_prefix(self):
        """Lines without a recognized speaker pattern become messages."""
        result = parse_chat_log("Just a random line of text")
        assert len(result) == 1

    def test_multiple_senders(self):
        log = "Alex: Hey\nBob: What's up\nYou: Nothing"
        result = parse_chat_log(log)
        names = [r["sender_name"] for r in result]
        assert "Alex" in names
        assert "Bob" in names
        assert "You" in names

    def test_suspect_partial_match(self):
        """Suspect name should match as substring."""
        log = "Alex_M: You're crazy\nYou: No"
        result = parse_chat_log(log, suspect_name="Alex")
        assert result[0]["sender"] == "suspect"
