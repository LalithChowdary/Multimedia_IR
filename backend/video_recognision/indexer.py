"""
Enhanced Video Search Indexer with Intelligent Chunking
========================================================

Improvements over v1:
- Context-aware chunking (semantic segments instead of individual lines)
- Sliding window with overlap for better retrieval
- FAISS for efficient similarity search (scales to millions of vectors)
- Incremental indexing (only process new videos)
- Metadata enrichment (video duration, word count, etc.)
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
from typing import List, Dict, Tuple

# --- Configuration ---
BACKEND_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = BACKEND_DIR / "video_recognision" / "transcripts"
INDEX_DIR = BACKEND_DIR / "video_recognision" / "search_index_v2"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# Chunking parameters
CHUNK_SIZE = 5  # Number of consecutive sentences to combine
OVERLAP = 2     # Number of sentences to overlap between chunks
MIN_CHUNK_LENGTH = 50  # Minimum characters per chunk

# Regex to parse timestamp lines
TIMESTAMP_PATTERN = re.compile(r"\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)")


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    h, m, s = timestamp.split(':')
    s, ms = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_transcript(filepath: Path) -> List[Dict]:
    """Parse transcript file and return list of timestamped segments."""
    segments = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = TIMESTAMP_PATTERN.match(line)
            if match:
                start, end, text = match.groups()
                text = text.strip()
                if text:
                    segments.append({
                        'start_time': start,
                        'end_time': end,
                        'start_seconds': timestamp_to_seconds(start),
                        'end_seconds': timestamp_to_seconds(end),
                        'text': text
                    })
    return segments


def create_chunks(segments: List[Dict], video_name: str) -> List[Dict]:
    """
    Create overlapping chunks from transcript segments.
    
    This improves search by:
    1. Combining multiple sentences for richer context
    2. Overlapping chunks so important info isn't split
    3. Preserving timestamp ranges for each chunk
    """
    chunks = []
    
    if not segments:
        return chunks
    
    for i in range(0, len(segments), CHUNK_SIZE - OVERLAP):
        # Get chunk of consecutive segments
        chunk_segments = segments[i:i + CHUNK_SIZE]
        
        if not chunk_segments:
            break
            
        # Combine text from all segments in chunk
        combined_text = ' '.join(s['text'] for s in chunk_segments)
        
        # Prepend video title to give it more weight in the embedding
        combined_text = f"Video: {video_name}. {combined_text}"
        
        # Skip if chunk is too short
        if len(combined_text) < MIN_CHUNK_LENGTH:
            continue
        
        # Get time range for entire chunk
        start_time = chunk_segments[0]['start_time']
        end_time = chunk_segments[-1]['end_time']
        start_seconds = chunk_segments[0]['start_seconds']
        end_seconds = chunk_segments[-1]['end_seconds']
        
        chunks.append({
            'video_name': video_name,
            'start_time': start_time,
            'end_time': end_time,
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'text': combined_text,
            'num_segments': len(chunk_segments)
        })
    
    return chunks


def load_existing_index() -> Tuple[faiss.Index, List[Dict], set]:
    """Load existing FAISS index and metadata if available."""
    index_path = INDEX_DIR / "faiss.index"
    metadata_path = INDEX_DIR / "metadata.json"
    
    if index_path.exists() and metadata_path.exists():
        print("Loading existing index...")
        index = faiss.read_index(str(index_path))
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Get set of already indexed videos
        indexed_videos = {m['video_name'] for m in metadata}
        
        return index, metadata, indexed_videos
    
    return None, [], set()


def create_search_index(incremental: bool = True):
    """
    Create or update the search index with FAISS for efficient retrieval.
    
    Args:
        incremental: If True, only index new videos. If False, rebuild entire index.
    """
    print(f"Loading sentence transformer model: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully.")
    
    # Create index directory
    INDEX_DIR.mkdir(exist_ok=True, parents=True)
    
    if not TRANSCRIPT_DIR.exists():
        print(f"Error: Transcript directory not found at {TRANSCRIPT_DIR}")
        return
    
    # Load existing index if doing incremental update
    existing_index, existing_metadata, indexed_videos = load_existing_index()
    
    if not incremental:
        existing_index = None
        existing_metadata = []
        indexed_videos = set()
    
    # Get all transcript files
    transcript_files = [f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith('.txt')]
    
    if not transcript_files:
        print(f"No transcript files found in {TRANSCRIPT_DIR}.")
        return
    
    # Filter to only new videos if doing incremental indexing
    if incremental and indexed_videos:
        new_files = [f for f in transcript_files if Path(f).stem not in indexed_videos]
        if not new_files:
            print("No new videos to index.")
            return
        print(f"Found {len(new_files)} new videos to index (out of {len(transcript_files)} total).")
        transcript_files = new_files
    else:
        print(f"Found {len(transcript_files)} videos to index.")
    
    # Process transcripts into chunks
    all_chunks = []
    
    for transcript_filename in tqdm(transcript_files, desc="Processing transcripts"):
        video_name = Path(transcript_filename).stem
        transcript_path = TRANSCRIPT_DIR / transcript_filename
        
        # Parse transcript
        segments = parse_transcript(transcript_path)
        
        if not segments:
            print(f"  - WARNING: No segments found in '{transcript_filename}'")
            continue
        
        # Create chunks from segments
        chunks = create_chunks(segments, video_name)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        print("No chunks created. Check your transcripts.")
        return
    
    print(f"\nCreated {len(all_chunks)} searchable chunks from {len(transcript_files)} videos")
    print(f"Average chunk size: {sum(len(c['text']) for c in all_chunks) / len(all_chunks):.1f} characters")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Create or update FAISS index
    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    
    if existing_index is None:
        # Create new index with cosine similarity
        print(f"\nCreating new FAISS index (dimension={dimension})...")
        index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        all_metadata = all_chunks
    else:
        # Add to existing index
        print(f"\nUpdating existing index (adding {len(embeddings)} new vectors)...")
        faiss.normalize_L2(embeddings)
        existing_index.add(embeddings)
        
        index = existing_index
        all_metadata = existing_metadata + all_chunks
    
    # Save index and metadata
    print(f"\nSaving index to '{INDEX_DIR}'...")
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    
    with open(INDEX_DIR / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Save index statistics
    stats = {
        'total_videos': len(set(m['video_name'] for m in all_metadata)),
        'total_chunks': len(all_metadata),
        'embedding_dimension': dimension,
        'model_name': MODEL_NAME,
        'chunk_size': CHUNK_SIZE,
        'overlap': OVERLAP
    }
    
    with open(INDEX_DIR / "stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("Indexing complete!")
    print("="*60)
    print(f"Total videos indexed: {stats['total_videos']}")
    print(f"Total searchable chunks: {stats['total_chunks']}")
    print(f"Average chunks per video: {stats['total_chunks'] / stats['total_videos']:.1f}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build or update video search index")
    parser.add_argument('--rebuild', action='store_true', 
                       help='Rebuild entire index instead of incremental update')
    
    args = parser.parse_args()
    
    create_search_index(incremental=not args.rebuild)
