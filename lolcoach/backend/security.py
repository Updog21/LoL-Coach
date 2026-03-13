"""Session token auth, keyring helpers, and replay protection."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import stat
import sys
from datetime import datetime, timezone

from fastapi import Header, HTTPException, Request

from .config import (
    KEYRING_SERVICE,
    SUPPORTED_PROVIDERS,
    TIMESTAMP_WINDOW_SECONDS,
    TOKEN_BYTES,
    TOKEN_FILE,
)

# ---------------------------------------------------------------------------
# Session token management
# ---------------------------------------------------------------------------
_cached_token: str | None = None


def _generate_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def get_or_create_token() -> str:
    """Read the session token from disk, or generate a new one."""
    global _cached_token
    if _cached_token is not None:
        return _cached_token

    if TOKEN_FILE.exists():
        _cached_token = TOKEN_FILE.read_text().strip()
    else:
        _cached_token = _generate_token()
        TOKEN_FILE.write_text(_cached_token)
        if sys.platform != "win32":
            TOKEN_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
    return _cached_token


def _constant_time_compare(a: str, b: str) -> bool:
    """SHA-256 then hmac.compare_digest to prevent timing oracle."""
    ha = hashlib.sha256(a.encode()).digest()
    hb = hashlib.sha256(b.encode()).digest()
    return hmac.compare_digest(ha, hb)


# ---------------------------------------------------------------------------
# FastAPI dependency — token validation
# ---------------------------------------------------------------------------
async def require_token(x_session_token: str = Header(...)) -> str:
    expected = get_or_create_token()
    if not _constant_time_compare(x_session_token, expected):
        raise HTTPException(status_code=401, detail="Invalid session token")
    return x_session_token


# ---------------------------------------------------------------------------
# Replay protection
# ---------------------------------------------------------------------------
def validate_timestamp(ts: datetime) -> None:
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = abs((now - ts).total_seconds())
    if delta > TIMESTAMP_WINDOW_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Timestamp outside {TIMESTAMP_WINDOW_SECONDS}s window",
        )


# ---------------------------------------------------------------------------
# Keyring helpers
# ---------------------------------------------------------------------------
def save_api_key(provider: str, key: str) -> None:
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    try:
        import keyring as kr

        kr.set_password(KEYRING_SERVICE, f"{provider}_api_key", key)
    except Exception:
        # Fallback for environments without a keyring backend (Linux dev, CI).
        # Store in a restricted file next to the token.
        _fallback_key_file(provider).write_text(key)
        if sys.platform != "win32":
            _fallback_key_file(provider).chmod(stat.S_IRUSR | stat.S_IWUSR)


def load_api_key(provider: str) -> str | None:
    if provider not in SUPPORTED_PROVIDERS:
        return None
    try:
        import keyring as kr

        val = kr.get_password(KEYRING_SERVICE, f"{provider}_api_key")
        if val:
            return val
    except Exception:
        pass
    fb = _fallback_key_file(provider)
    if fb.exists():
        return fb.read_text().strip()
    return None


def delete_api_key(provider: str) -> None:
    try:
        import keyring as kr

        kr.delete_password(KEYRING_SERVICE, f"{provider}_api_key")
    except Exception:
        pass
    fb = _fallback_key_file(provider)
    if fb.exists():
        fb.unlink()


def _fallback_key_file(provider: str):
    from .config import DATA_DIR

    return DATA_DIR / f".key_{provider}"
