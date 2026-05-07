import os
from typing import List
from google import genai
from src.core.schema import EmbeddingResult
import numpy as np

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def get_embedding(text: str) -> EmbeddingResult:
    """
    Get text embedding using Gemini's embedding model.
    Note: Requires GEMINI_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please configure it via '/config'")
    
    client = genai.Client(api_key=api_key)
    
    # Use symmetric task formatting for semantic similarity
    formatted_text = f"task: sentence similarity | query: {text}"
    
    try:
        result = client.models.embed_content(
            model="gemini-embedding-2",
            contents=formatted_text,
        )
        vector = result.embeddings[0].values
        tokens_used = len(formatted_text) // 4
        return EmbeddingResult(vector=vector, tokens_used=tokens_used)
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {str(e)}")

def calculate_group_consistency(texts: List[str]) -> float:
    """
    Calculate semantic consistency (0 to 1) for a group of texts.
    1.0 means identical semantics, 0.0 means completely orthogonal/different.
    """
    if len(texts) <= 1:
        return 1.0
        
    embeddings = [get_embedding(t).vector for t in texts]
    
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
            
    # Average pair-wise similarity
    return float(np.mean(similarities))
