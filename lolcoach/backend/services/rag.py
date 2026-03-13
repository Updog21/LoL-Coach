"""ChromaDB ingest + retrieval for per-champion custom resources."""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import CHUNK_OVERLAP, CHUNK_SIZE, CHROMA_DIR, RAG_TOP_K
from .ai_router import _get_embedding_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChromaDB client (singleton)
# ---------------------------------------------------------------------------
_client: chromadb.ClientAPI | None = None


def _get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _collection_name(champion: str) -> str:
    """Sanitized collection name for a champion."""
    return f"champ_{champion.lower().strip()}"


# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
async def ingest_text(
    champion: str,
    text: str,
    source_url: str,
    label: str,
    resource_id: int,
    provider: str = "openai",
) -> int:
    """Split text into chunks, embed, and store in ChromaDB. Returns chunk count."""
    chunks = _splitter.split_text(text)
    if not chunks:
        return 0

    client = _get_chroma_client()
    collection = client.get_or_create_collection(name=_collection_name(champion))

    embedding_fn = _get_embedding_model(provider)

    ids = [f"res{resource_id}_chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source_url": source_url or "",
            "label": label,
            "champion": champion.lower(),
            "resource_id": resource_id,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    if embedding_fn:
        embeddings = embedding_fn.embed_documents(chunks)
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    else:
        # Use ChromaDB's default embedding function
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    logger.info("Ingested %d chunks for %s (resource %d)", len(chunks), champion, resource_id)
    return len(chunks)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
async def retrieve_context(
    champion: str,
    query: str,
    provider: str = "openai",
    top_k: int = RAG_TOP_K,
) -> str:
    """Retrieve top-k relevant chunks for a champion + query. Returns formatted text."""
    client = _get_chroma_client()
    col_name = _collection_name(champion)

    try:
        collection = client.get_collection(name=col_name)
    except Exception:
        return ""

    embedding_fn = _get_embedding_model(provider)

    if embedding_fn:
        query_embedding = embedding_fn.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    else:
        results = collection.query(query_texts=[query], n_results=top_k)

    if not results["documents"] or not results["documents"][0]:
        return ""

    parts: list[str] = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source = meta.get("label", meta.get("source_url", "unknown"))
        parts.append(f"[Source: {source}]\n{doc}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------
async def delete_resource_chunks(champion: str, resource_id: int) -> None:
    """Remove all chunks for a specific resource from ChromaDB."""
    client = _get_chroma_client()
    col_name = _collection_name(champion)

    try:
        collection = client.get_collection(name=col_name)
    except Exception:
        return

    collection.delete(where={"resource_id": resource_id})
    logger.info("Deleted chunks for resource %d from %s", resource_id, champion)
