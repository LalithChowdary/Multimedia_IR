import os
import json
import time
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

# --- Configuration ---
BACKEND_DIR = Path(__file__).resolve().parent
INDEX_DIR = BACKEND_DIR / "search_index_v2"
METADATA_FILE = INDEX_DIR / "metadata.json"
INDEX_FILE = INDEX_DIR / "faiss.index"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- Ground Truth Data ---
# Format: {'query': "...", 'expected_partial': "..."}
# expected_partial: A unique substring of the video filename to match
GROUND_TRUTH = [
    # Video: iPhone 17 Pro Max Review
    {"query": "iPhone 17 Pro Max battery life", "expected_partial": "iPhone 17 Pro Max Review"},
    {"query": "new ghost case from Debrand", "expected_partial": "iPhone 17 Pro Max Review"},
    {"query": "120Hz promotion panel", "expected_partial": "iPhone 17 Pro Max Review"},
    
    # Video: Biggest Upgrade Scam of 2025 (Hindi/English)
    {"query": "Phoebus cartel lightbulb", "expected_partial": "Biggest Upgrade Scam"},
    {"query": "planned obsolescence scam", "expected_partial": "Biggest Upgrade Scam"},
    {"query": "lightbulb conspiracy", "expected_partial": "Biggest Upgrade Scam"},
    
    # Video: iPhone 17 / Pro / Air
    {"query": "iPhone Air thickness", "expected_partial": "iPhone 17 ⧸ Pro ⧸ Air"},
    {"query": "A19 Pro chip single camera", "expected_partial": "iPhone 17 ⧸ Pro ⧸ Air"},
    {"query": "titanium frame ceramic glass", "expected_partial": "iPhone 17 ⧸ Pro ⧸ Air"},

    # Video: Indian Cities
    {"query": "why Indian cities suck", "expected_partial": "I investigated why Indian cities suck"},
    {"query": "urban planning issues india", "expected_partial": "I investigated why Indian cities suck"},

    # Video: Radix Sort
    {"query": "radix sort algorithm explanation", "expected_partial": "7.10 Radix Sort"},
    {"query": "bucket sort data structure", "expected_partial": "7.10 Radix Sort"},
]

def load_resources():
    print("Loading resources...")
    # Load Metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    # Load FAISS Index
    index = faiss.read_index(str(INDEX_FILE))
    
    # Load Model
    model = SentenceTransformer(MODEL_NAME)
    
    return metadata, index, model

def evaluate_semantic_search(queries, metadata, index, model, top_k=5):
    print(f"\n--- Evaluating Semantic Search (FAISS) ---")
    correct_top1 = 0
    correct_top5 = 0
    total_time = 0
    
    results = []

    for item in queries:
        query = item['query']
        expected = item['expected_partial']
        
        start_time = time.time()
        
        # 1. Encode
        query_vector = model.encode([query])
        faiss.normalize_L2(query_vector)
        
        # 2. Search
        distances, indices = index.search(query_vector, top_k)
        
        end_time = time.time()
        total_time += (end_time - start_time)
        
        # 3. Check Results
        found_top1 = False
        found_top5 = False
        
        retrieved_videos = []
        for idx in indices[0]:
            if idx < len(metadata):
                vid_name = metadata[idx]['video_name']
                retrieved_videos.append(vid_name)
        
        # Check Top-1
        if retrieved_videos and expected in retrieved_videos[0]:
            correct_top1 += 1
            found_top1 = True
            
        # Check Top-5
        for vid in retrieved_videos:
            if expected in vid:
                correct_top5 += 1
                found_top5 = True
                break
        
        results.append({
            "query": query,
            "found_top1": found_top1,
            "found_top5": found_top5,
            "top_result": retrieved_videos[0] if retrieved_videos else "None"
        })
        
    avg_time = (total_time / len(queries)) * 1000 # ms
    acc_top1 = (correct_top1 / len(queries)) * 100
    acc_top5 = (correct_top5 / len(queries)) * 100
    
    print(f"Semantic Search Results:")
    print(f"Top-1 Accuracy: {acc_top1:.2f}%")
    print(f"Top-5 Accuracy: {acc_top5:.2f}%")
    print(f"Avg Latency:    {avg_time:.2f} ms")
    
    return acc_top1, acc_top5, avg_time

def evaluate_tfidf_baseline(queries, metadata):
    print(f"\n--- Evaluating TF-IDF Baseline ---")
    
    # 1. Prepare Corpus (Group chunks by video for fair document-level comparison)
    # Actually, to be comparable to Semantic Search which indexes CHUNKS, 
    # we should index CHUNKS for TF-IDF too.
    
    corpus = [m['text'] for m in metadata]
    video_names = [m['video_name'] for m in metadata]
    
    # 2. Build TF-IDF Matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    correct_top1 = 0
    correct_top5 = 0
    total_time = 0
    
    for item in queries:
        query = item['query']
        expected = item['expected_partial']
        
        start_time = time.time()
        
        # 3. Transform Query
        query_vec = vectorizer.transform([query])
        
        # 4. Cosine Similarity
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # 5. Get Top-K
        top_k_indices = similarities.argsort()[-5:][::-1]
        
        end_time = time.time()
        total_time += (end_time - start_time)
        
        retrieved_videos = [video_names[i] for i in top_k_indices]
        
        # Check Top-1
        if retrieved_videos and expected in retrieved_videos[0]:
            correct_top1 += 1
            
        # Check Top-5
        found_top5 = False
        for vid in retrieved_videos:
            if expected in vid:
                found_top5 = True
                break
        if found_top5:
            correct_top5 += 1

    avg_time = (total_time / len(queries)) * 1000
    acc_top1 = (correct_top1 / len(queries)) * 100
    acc_top5 = (correct_top5 / len(queries)) * 100
    
    print(f"TF-IDF Baseline Results:")
    print(f"Top-1 Accuracy: {acc_top1:.2f}%")
    print(f"Top-5 Accuracy: {acc_top5:.2f}%")
    print(f"Avg Latency:    {avg_time:.2f} ms")
    
    return acc_top1, acc_top5, avg_time

if __name__ == "__main__":
    metadata, index, model = load_resources()
    
    print(f"Evaluating on {len(GROUND_TRUTH)} queries...")
    
    s_acc1, s_acc5, s_time = evaluate_semantic_search(GROUND_TRUTH, metadata, index, model)
    t_acc1, t_acc5, t_time = evaluate_tfidf_baseline(GROUND_TRUTH, metadata)
    
    print("\n" + "="*40)
    print("FINAL COMPARISON")
    print("="*40)
    print(f"{'Metric':<15} | {'Semantic':<10} | {'TF-IDF':<10}")
    print("-" * 40)
    print(f"{'Top-1 Acc':<15} | {s_acc1:.1f}%      | {t_acc1:.1f}%")
    print(f"{'Top-5 Acc':<15} | {s_acc5:.1f}%      | {t_acc5:.1f}%")
    print(f"{'Latency (ms)':<15} | {s_time:.1f}       | {t_time:.1f}")
    print("="*40)
