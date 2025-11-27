
import sys
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Setup paths
BACKEND_DIR = Path("backend").resolve()
INDEX_DIR = BACKEND_DIR / "video_recognision" / "search_index_v2"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

def debug_search():
    query = "तर्टीस्ट्खट्चीबी डेटा, अगर अब आवरेज़ग, भरत्टीस्टचीबी डेटा इस्टमाल करते है"
    print(f"Query: {query}")

    # Load metadata
    metadata_path = INDEX_DIR / "metadata.json"
    if not metadata_path.exists():
        print("Metadata not found!")
        return

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Find the target chunk in metadata
    target_video_partial = "iQOO 15 Price"
    target_chunk_idx = -1
    target_chunk_text = ""
    
    print(f"\nSearching metadata for video containing '{target_video_partial}' and query text...")
    
    # We need to look for the unicode escaped version or just string match
    # The query in the file matches exactly what we have.
    
    for i, meta in enumerate(metadata):
        if target_video_partial in meta['video_name']:
            # Check if query text is in this chunk's text
            if query in meta['text']:
                print(f"Found target chunk at index {i}!")
                print(f"Video Name: {meta['video_name']}")
                print(f"Full Chunk Text: {meta['text']}")
                target_chunk_idx = i
                target_chunk_text = meta['text']
                break
    
    if target_chunk_idx == -1:
        print("Could not find the specific text chunk in metadata! This is the root cause if true.")
        # Let's try to find just the video to see if it exists at all
        found_video = False
        for meta in metadata:
             if target_video_partial in meta['video_name']:
                 found_video = True
        if found_video:
            print("Video exists in metadata, but exact query text was not found in any chunk.")
        else:
            print("Video does not exist in metadata.")
        return

    # Load Index
    index_path = INDEX_DIR / "faiss.index"
    index = faiss.read_index(str(index_path))
    
    # Load Model
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Embed Query
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # Search
    k = 20
    distances, indices = index.search(query_embedding, k)
    
    print(f"\nTop {k} Results:")
    found_in_top_k = False
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
        if idx == -1: continue
        meta = metadata[idx]
        is_target = (idx == target_chunk_idx)
        marker = "*** TARGET ***" if is_target else ""
        if is_target: found_in_top_k = True
        
        print(f"{rank+1}. Score: {score:.4f} | Video: {meta['video_name'][:30]}... {marker}")
        if is_target:
            print(f"   -> Target Text: {meta['text'][:100]}...")

    if not found_in_top_k:
        print(f"\nTarget chunk (Index {target_chunk_idx}) was NOT in top {k}.")
        
        # Calculate manual score
        # We can't easily get the vector from the index unless we reconstruct it or if the index supports it.
        # IndexFlatIP stores vectors, so we can reconstruct.
        try:
            target_vector = index.reconstruct(target_chunk_idx)
            # Reshape for dot product
            target_vector = target_vector.reshape(1, -1)
            manual_score = np.dot(query_embedding, target_vector.T)[0][0]
            print(f"Manual Score for Target Chunk (with Title): {manual_score:.4f}")
        except Exception as e:
            print(f"Could not reconstruct vector: {e}")


if __name__ == "__main__":
    debug_search()
