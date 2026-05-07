import os
import hashlib
import logging
from typing import Dict, List
from google import genai
from src.core.schema import EmbeddingResult, ConsistencyResult
import numpy as np

logger = logging.getLogger(__name__)

# In-memory embedding cache keyed by SHA-256 of input text.
# Reset each session (no persistence).
_embedding_cache: Dict[str, List[float]] = {}


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Returns 0.0 if either vector has zero magnitude (guard against
    NaN / ZeroDivisionError from degenerate embeddings).
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def get_embedding(text: str) -> EmbeddingResult:
    """Get text embedding using Gemini's embedding model.

    Results are cached by content hash to avoid redundant API calls
    when candidates share identical text.

    Note: Requires GEMINI_API_KEY environment variable to be set.
    """
    # Check cache first
    cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if cache_key in _embedding_cache:
        logger.debug("Embedding cache hit for %s…", cache_key[:12])
        return EmbeddingResult(vector=_embedding_cache[cache_key], tokens_used=0)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please configure it via '/config'")
    
    client = genai.Client(api_key=api_key)
    
    formatted_text = f"task: sentence similarity | query: {text}"
    
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=formatted_text,
        )
        vector = result.embeddings[0].values
        tokens_used = len(formatted_text) // 4
        # Store in cache
        _embedding_cache[cache_key] = vector
        return EmbeddingResult(vector=vector, tokens_used=tokens_used)
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {str(e)}")

def calculate_group_consistency(texts: List[str]) -> ConsistencyResult:
    """Calculate semantic consistency and return embeddings for reuse."""
    if len(texts) <= 1:
        return ConsistencyResult(score=1.0)

    embeddings = [get_embedding(t).vector for t in texts]

    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarities.append(cosine_similarity(embeddings[i], embeddings[j]))

    score = float(np.mean(similarities))
    rep_idx = select_representative_index(embeddings)
    min_idx = select_minority_index(embeddings, rep_idx)

    return ConsistencyResult(
        score=score,
        embeddings=embeddings,
        representative_idx=rep_idx,
        minority_idx=min_idx,
    )

def select_representative_index(embeddings: List[List[float]]) -> int:
    """Select the candidate closest to the group centroid."""
    n = len(embeddings)
    if n <= 1:
        return 0
    best_idx, best_avg = 0, -1.0
    for i in range(n):
        avg = sum(
            cosine_similarity(embeddings[i], embeddings[j])
            for j in range(n) if j != i
        ) / (n - 1)
        if avg > best_avg:
            best_avg = avg
            best_idx = i
    return best_idx

def select_minority_index(embeddings: List[List[float]], representative_idx: int) -> int:
    """Select the candidate farthest from the representative."""
    if len(embeddings) <= 1:
        return 0
    rep = embeddings[representative_idx]
    worst_idx, worst_sim = 0, 2.0
    for i, emb in enumerate(embeddings):
        if i == representative_idx:
            continue
        sim = cosine_similarity(rep, emb)
        if sim < worst_sim:
            worst_sim = sim
            worst_idx = i
    return worst_idx
