from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import json

import numpy as np

from .types import ChunkRecord, RetrievalHit

try:
    from fastembed import TextEmbedding
except ImportError:  # pragma: no cover - dependency installed at runtime
    TextEmbedding = None


_EMBEDDER_CACHE: dict[str, Any] = {}


def get_text_embedder(model_name: str) -> Any:
    if TextEmbedding is None:
        raise RuntimeError(
            "fastembed is required for local dense retrieval. Install the dependencies from requirements.txt first."
        )
    if model_name not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[model_name] = TextEmbedding(model_name=model_name)
    return _EMBEDDER_CACHE[model_name]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_dense_embeddings(chunks: list[ChunkRecord], model_name: str) -> np.ndarray:
    embedder = get_text_embedder(model_name)
    texts = [chunk.text for chunk in chunks]
    passage_vectors = list(embedder.passage_embed(texts))
    matrix = np.asarray(passage_vectors, dtype=np.float32)
    return l2_normalize(matrix)


def embed_query(query: str, model_name: str) -> np.ndarray:
    embedder = get_text_embedder(model_name)
    query_vector = np.asarray(list(embedder.query_embed([query]))[0], dtype=np.float32).reshape(1, -1)
    return l2_normalize(query_vector)[0]


def save_dense_index(base_dir: Path, *, embeddings: np.ndarray, chunks: list[ChunkRecord], model_name: str) -> dict[str, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = base_dir / "dense_embeddings.npy"
    metadata_path = base_dir / "dense_index_metadata.json"

    np.save(embeddings_path, embeddings)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "chunk_count": len(chunks),
                "embedding_dimension": int(embeddings.shape[1]) if embeddings.size else 0,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
    }


def load_dense_index(base_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    embeddings_path = base_dir / "dense_embeddings.npy"
    metadata_path = base_dir / "dense_index_metadata.json"
    embeddings = np.load(embeddings_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return embeddings, metadata


def dense_retrieve(
    query: str,
    *,
    chunks: list[ChunkRecord],
    embeddings: np.ndarray,
    model_name: str,
    top_k: int,
) -> list[RetrievalHit]:
    if not chunks:
        return []
    query_vector = embed_query(query, model_name=model_name)
    scores = embeddings @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]

    hits: list[RetrievalHit] = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[int(idx)]
        hits.append(
            RetrievalHit(
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
                rank=rank,
                score=float(round(float(scores[int(idx)]), 6)),
                citation_label=chunk.citation_label,
                excerpt=chunk.text[:400].strip(),
                metadata=chunk.metadata,
            )
        )
    return hits


def reciprocal_rank_fusion(
    hit_groups: list[list[RetrievalHit]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievalHit]:
    fused_scores: dict[str, float] = defaultdict(float)
    representative_hits: dict[str, RetrievalHit] = {}

    for hits in hit_groups:
        for rank, hit in enumerate(hits, start=1):
            fused_scores[hit.chunk_id] += 1.0 / (rrf_k + rank)
            representative_hits.setdefault(hit.chunk_id, hit)

    ranked_chunk_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)[:top_k]
    fused_hits: list[RetrievalHit] = []
    for fused_rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        base_hit = representative_hits[chunk_id]
        fused_hits.append(
            RetrievalHit(
                source_id=base_hit.source_id,
                chunk_id=base_hit.chunk_id,
                rank=fused_rank,
                score=round(fused_scores[chunk_id], 6),
                citation_label=base_hit.citation_label,
                excerpt=base_hit.excerpt,
                metadata={
                    **base_hit.metadata,
                    "fusion_method": "rrf",
                },
            )
        )
    return fused_hits
