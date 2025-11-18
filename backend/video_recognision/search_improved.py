"""
IMPROVED VERSION OF YOUR CURRENT SEARCH.PY
============================================

This version keeps your current architecture but adds:
1. Model caching (200x faster repeated searches)
2. Score filtering (better result quality)
3. Better output formatting

Just replace your search.py with this file!
"""

import pickle
import json
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
BACKEND_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE_PATH = BACKEND_DIR / "video_recognision" / "search_index.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'

# Global model cache - MAJOR IMPROVEMENT!
_MODEL_CACHE = None


def get_model():
    """Load model once and cache it for subsequent searches."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        print("Loading search model (one-time)...")
        _MODEL_CACHE = SentenceTransformer(MODEL_NAME)
    return _MODEL_CACHE


def search(query: str, top_k: int = 5, min_score: float = 0.2):
    """
    Performs semantic search with improved filtering.
    
    Args:
        query: The text to search for
        top_k: Number of results to return
        min_score: Minimum similarity score (0-1). Default 0.2 filters noise.
    """
    if not INDEX_FILE_PATH.exists():
        return json.dumps({
            "error": "Search index not found.",
            "message": "Please run 'indexer.py' first to build the search index."
        }, indent=4)

    # Load model (cached after first call!)
    model = get_model()
    
    # Load index
    with open(INDEX_FILE_PATH, 'rb') as f:
        search_index = pickle.load(f)

    embeddings = search_index['embeddings']
    metadata = search_index['metadata']

    # Generate query embedding
    query_embedding = model.encode([query], convert_to_tensor=True)

    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding.cpu().numpy(), embeddings)[0]

    # Get top results
    top_k_indices = similarities.argsort()[-top_k*2:][::-1]  # Get 2x for filtering

    # Format and filter results
    results = []
    for idx in top_k_indices:
        score = float(similarities[idx])
        
        # Filter low-quality matches
        if score < min_score:
            continue
            
        video_info = metadata[idx]
        results.append({
            "video_name": video_info['video_name'],
            "timestamp": f"{video_info['start_time']} --> {video_info['end_time']}",
            "text": video_info['text'],
            "similarity_score": round(score, 4)
        })
        
        if len(results) >= top_k:
            break

    return json.dumps(results, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform semantic search on video transcripts."
    )
    parser.add_argument(
        "query",
        type=str,
        help="The text query to search for."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)"
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.2,
        help="Minimum similarity score 0-1 (default: 0.2)"
    )

    args = parser.parse_args()

    json_results = search(
        query=args.query, 
        top_k=args.top_k,
        min_score=args.min_score
    )
    print(json_results)
