"""
Encryption utilities for securing sensitive context state data.
Uses Fernet symmetric encryption from the cryptography library.
Falls back to plaintext JSON if cryptography is not installed.
"""
import json
import os
import logging
import base64
import hashlib

logger = logging.getLogger(__name__)

# Try to import cryptography; gracefully degrade if unavailable
try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("cryptography package not installed. Context state will NOT be encrypted.")


def _get_key() -> bytes:
    """
    Derive an encryption key from the MANTACAI_SECRET_KEY env var.
    If not set, generates a deterministic key from the machine's hostname
    (NOT secure for production — set the env var in production).
    """
    secret = os.environ.get("MANTACAI_SECRET_KEY")
    if secret:
        # Derive a 32-byte key from the secret using SHA-256
        key_bytes = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(key_bytes)
    else:
        logger.warning(
            "MANTACAI_SECRET_KEY not set. Using machine-derived key. "
            "Set this env var for production use."
        )
        fallback = hashlib.sha256(f"mantacai-{os.name}".encode()).digest()
        return base64.urlsafe_b64encode(fallback)


def save_encrypted(filepath: str, data: dict) -> None:
    """Save data as encrypted JSON file."""
    json_bytes = json.dumps(data).encode("utf-8")

    if HAS_CRYPTO:
        fernet = Fernet(_get_key())
        encrypted = fernet.encrypt(json_bytes)
        with open(filepath, "wb") as f:
            f.write(encrypted)
    else:
        # Fallback: plaintext
        with open(filepath, "w") as f:
            json.dump(data, f)


def load_encrypted(filepath: str) -> dict:
    """Load data from encrypted JSON file."""
    if not os.path.exists(filepath):
        return {}

    if HAS_CRYPTO:
        fernet = Fernet(_get_key())
        try:
            with open(filepath, "rb") as f:
                encrypted = f.read()
            decrypted = fernet.decrypt(encrypted)
            return json.loads(decrypted.decode("utf-8"))
        except Exception:
            # If decryption fails (old plaintext file), try loading as JSON
            logger.warning("Decryption failed, trying plaintext fallback.")
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
    else:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {}
