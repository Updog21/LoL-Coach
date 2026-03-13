"""Paths, tunables, and constants for LoL Coach."""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directories — portable: everything lives next to the executable
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    # Running as PyInstaller bundle — sys.executable is the .exe path
    _EXE_DIR = Path(sys.executable).resolve().parent
else:
    # Running from source — use the lolcoach/ package directory
    _EXE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = _EXE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8765
CORS_ORIGIN = f"http://{HOST}:{PORT}"

# ---------------------------------------------------------------------------
# Token / auth
# ---------------------------------------------------------------------------
TOKEN_FILE = DATA_DIR / ".token"
TOKEN_BYTES = 32  # 32 bytes → 43-char URL-safe base64

# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
DB_PATH = DATA_DIR / "lolcoach.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_DIR = DATA_DIR / "chroma"

# ---------------------------------------------------------------------------
# Riot Live Client API
# ---------------------------------------------------------------------------
RIOT_API_BASE = "https://127.0.0.1:2999"
RIOT_POLL_INTERVAL = 5.0  # seconds

# ---------------------------------------------------------------------------
# AI defaults
# ---------------------------------------------------------------------------
DEFAULT_PROVIDER = "openai"
SUPPORTED_PROVIDERS = ("openai", "gemini", "anthropic", "ollama")
OLLAMA_BASE_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# RAG tuning
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RAG_TOP_K = 5

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
ASK_RATE_LIMIT = "10/minute"
RESOURCE_RATE_LIMIT = "5/minute"
META_REFRESH_RATE_LIMIT = "2/hour"
DEFAULT_RATE_LIMIT = "60/minute"

# ---------------------------------------------------------------------------
# Replay protection
# ---------------------------------------------------------------------------
TIMESTAMP_WINDOW_SECONDS = 30

# ---------------------------------------------------------------------------
# Keyring service name
# ---------------------------------------------------------------------------
KEYRING_SERVICE = "lolcoach"
