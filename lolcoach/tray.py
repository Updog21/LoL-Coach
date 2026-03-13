"""Entry point — system tray icon + uvicorn server thread.

Portable: no installation required. Run the .exe from any folder.
All data (DB, tokens, ChromaDB) is stored in a 'data/' folder next to the executable.
"""

from __future__ import annotations

import logging
import sys
import threading
import webbrowser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("lolcoach")


def _run_server() -> None:
    """Run the FastAPI server in a background thread."""
    import uvicorn

    from backend.config import HOST, PORT

    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        log_level="warning",
        access_log=False,
    )


def _open_browser() -> None:
    from backend.config import HOST, PORT

    webbrowser.open(f"http://{HOST}:{PORT}/?autoAuth=1")


def main() -> None:
    from backend.config import DATA_DIR

    logger.info("Data directory: %s", DATA_DIR)

    # Start server thread
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()
    logger.info("Server starting on http://127.0.0.1:8765")

    try:
        import pystray
        from PIL import Image

        icon_image = Image.new("RGB", (64, 64), color=(30, 80, 180))

        def on_open(icon, item):
            _open_browser()

        def on_quit(icon, item):
            icon.stop()

        icon = pystray.Icon(
            "lolcoach",
            icon_image,
            "LoL Build Coach",
            menu=pystray.Menu(
                pystray.MenuItem("Open Dashboard", on_open, default=True),
                pystray.MenuItem("Quit", on_quit),
            ),
        )

        threading.Timer(1.5, _open_browser).start()
        icon.run()
    except ImportError:
        logger.info("No tray support — running headless. Press Ctrl+C to quit.")
        _open_browser()
        try:
            server_thread.join()
        except KeyboardInterrupt:
            logger.info("Shutting down.")
            sys.exit(0)


if __name__ == "__main__":
    main()
