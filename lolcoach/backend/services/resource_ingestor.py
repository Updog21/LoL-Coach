"""Dispatch resource ingestion by type — YouTube, article, PDF, note."""

from __future__ import annotations

import logging
import tempfile

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


async def fetch_youtube_transcript(url: str) -> str:
    """Fetch transcript text from a YouTube video URL."""
    from youtube_transcript_api import YouTubeTranscriptApi

    # Extract video ID from URL
    video_id = None
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]

    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url}")

    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join(entry["text"] for entry in transcript_list)


async def fetch_article_text(url: str) -> str:
    """Fetch and extract readable text from an article URL."""
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script/style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract text from article body or main content
    article = soup.find("article") or soup.find("main") or soup.find("body")
    if article:
        return article.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


async def fetch_pdf_text(url: str) -> str:
    """Download and extract text from a PDF URL."""
    import fitz  # PyMuPDF

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(resp.content)
        tmp.flush()
        doc = fitz.open(tmp.name)
        text_parts = [page.get_text() for page in doc]
        doc.close()

    return "\n".join(text_parts)


async def extract_text(resource_type: str, url: str | None, content: str | None) -> str:
    """Dispatch to the appropriate extractor based on resource type."""
    if resource_type == "note":
        return content or ""

    if not url:
        raise ValueError("URL required for non-note resources")

    if resource_type == "youtube":
        return await fetch_youtube_transcript(url)
    if resource_type == "article":
        return await fetch_article_text(url)
    if resource_type == "pdf":
        return await fetch_pdf_text(url)

    raise ValueError(f"Unknown resource type: {resource_type}")
