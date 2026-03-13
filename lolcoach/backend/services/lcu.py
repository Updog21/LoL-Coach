"""League Client (LCU) connector — lockfile discovery + authenticated requests."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

import httpx

from ..config import LCU_LOCKFILE_PATHS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lockfile parsing
# ---------------------------------------------------------------------------
# Lockfile format: "LeagueClient:PID:PORT:TOKEN:PROTOCOL"
# Example:         "LeagueClient:12345:52743:abc123def:https"

_LOCKFILE_RE = re.compile(r"^LeagueClient:(\d+):(\d+):(.+):(https?)$")


class LCUConnection:
    """Holds the LCU connection details parsed from lockfile."""

    def __init__(self, pid: int, port: int, token: str, protocol: str) -> None:
        self.pid = pid
        self.port = port
        self.token = token
        self.protocol = protocol
        self.base_url = f"{protocol}://127.0.0.1:{port}"
        # LCU uses Basic auth with username "riot" and the token as password
        creds = base64.b64encode(f"riot:{token}".encode()).decode()
        self.auth_header = f"Basic {creds}"

    def __repr__(self) -> str:
        return f"LCUConnection(port={self.port}, pid={self.pid})"


def find_lockfile() -> LCUConnection | None:
    """Search common install paths for the League Client lockfile."""
    for path_str in LCU_LOCKFILE_PATHS:
        p = Path(path_str)
        if p.exists():
            try:
                text = p.read_text().strip()
                m = _LOCKFILE_RE.match(text)
                if m:
                    conn = LCUConnection(
                        pid=int(m.group(1)),
                        port=int(m.group(2)),
                        token=m.group(3),
                        protocol=m.group(4),
                    )
                    logger.info("Found LCU lockfile: %s → port %d", path_str, conn.port)
                    return conn
            except Exception as exc:
                logger.debug("Failed to read lockfile %s: %s", path_str, exc)
                continue

    # Fallback: scan running processes for --app-port argument (Windows)
    return _find_lockfile_from_process()


def _find_lockfile_from_process() -> LCUConnection | None:
    """Fallback: find LCU port/token from LeagueClientUx process command line."""
    try:
        import subprocess

        result = subprocess.run(
            ["wmic", "process", "where", "name='LeagueClientUx.exe'",
             "get", "CommandLine", "/value"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout

        port_match = re.search(r"--app-port=(\d+)", output)
        token_match = re.search(r"--remoting-auth-token=([\w-]+)", output)

        if port_match and token_match:
            conn = LCUConnection(
                pid=0,
                port=int(port_match.group(1)),
                token=token_match.group(1),
                protocol="https",
            )
            logger.info("Found LCU from process: port %d", conn.port)
            return conn
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Authenticated HTTP client for LCU
# ---------------------------------------------------------------------------
async def lcu_get(conn: LCUConnection, endpoint: str) -> dict | list | None:
    """Make an authenticated GET request to the LCU API."""
    async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
        resp = await client.get(
            f"{conn.base_url}{endpoint}",
            headers={"Authorization": conn.auth_header, "Accept": "application/json"},
        )
        if resp.status_code == 404:
            return None  # Endpoint not available (e.g., not in champ select)
        resp.raise_for_status()
        return resp.json()
