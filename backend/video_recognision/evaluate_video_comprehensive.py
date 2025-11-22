"""
Comprehensive Video Search Evaluation for Research Paper
Generates multiple metrics, comparisons, and visualizations
"""
import os
import json
import time
import csv
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# --- Configuration ---
BACKEND_DIR = Path(__file__).resolve().parent
INDEX_DIR = BACKEND_DIR / "search_index_v2"
RESULTS_DIR = BACKEND_DIR / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)

METADATA_FILE = INDEX_DIR / "metadata.json"
INDEX_FILE = INDEX_DIR / "faiss.index"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# Expanded Ground Truth (20 queries)
GROUND_TRUTH = [
    # English → English (exact titles)
    {"query": "iPhone 17 Pro Max Review", "expected_partial": "iPhone 17 Pro Max Review", "lang": "EN→EN", "relevance": 2},
    {"query": "Dope Tech", "expected_partial": "Dope Tech", "lang": "EN→EN", "relevance": 2},
    
    # English → English (content)
    {"query": "iPhone battery life improvements", "expected_partial": "iPhone 17 Pro Max Review", "lang": "EN→EN", "relevance": 2},
    {"query": "120Hz promotion display panel", "expected_partial": "iPhone 17 Pro Max Review", "lang": "EN→EN", "relevance": 2},
    {"query": "ceramic shield glass protection", "expected_partial": "iPhone 17 Pro Max Review", "lang": "EN→EN", "relevance": 1},
    {"query": "unibody aluminum design titanium", "expected_partial": "iPhone 17 ⧸ Pro ⧸ Air", "lang": "EN→EN", "relevance": 2},
    {"query": "iPhone Air thickness single camera", "expected_partial": "iPhone 17 ⧸ Pro ⧸ Air", "lang": "EN→EN", "relevance": 2},
    {"query": "video camera comparison pixel", "expected_partial": "iPhone Video Is Still Better", "lang": "EN→EN", "relevance": 1},
    
    # English → Hindi (cross-language)
    {"query": "Phoebus cartel lightbulb conspiracy", "expected_partial": "Biggest Upgrade Scam", "lang": "EN→HI", "relevance": 2},
    {"query": "planned obsolescence smartphones", "expected_partial": "Biggest Upgrade Scam", "lang": "EN→HI", "relevance": 2},
    {"query": "iQOO 15 price specifications", "expected_partial": "iQOO 15 Price", "lang": "EN→HI", "relevance": 2},
    
    # Hindi → Hindi
    {"query": "आईफोन की कीमत", "expected_partial": "iQOO 15 Price", "lang": "HI→HI", "relevance": 1},
    
    # Technical content
    {"query": "radix sort bucket sort algorithm", "expected_partial": "7.10 Radix Sort", "lang": "EN→EN", "relevance": 2},
    {"query": "data structure sorting technique", "expected_partial": "7.10 Radix Sort", "lang": "EN→EN", "relevance": 1},
    {"query": "urban planning infrastructure problems", "expected_partial": "I investigated why Indian cities suck", "lang": "EN→EN", "relevance": 2},
    {"query": "Indian city development issues", "expected_partial": "I investigated why Indian cities suck", "lang": "EN→EN", "relevance": 2},
    
    # Product reviews
    {"query": "MacBook Pro M5 chip", "expected_partial": "Is Apple Bored of Winning", "lang": "EN→EN", "relevance": 2},
    {"query": "AirPods Pro 3 features", "expected_partial": "AirPods Pro 3 Review", "lang": "EN→EN", "relevance": 2},
    {"query": "gaming laptop professionals", "expected_partial": "The Gaming Laptop for Pros", "lang": "EN→EN", "relevance": 1},
    {"query": "wired Mac usage weird", "expected_partial": "The WEiRDEST Way to Use a Mac", "lang": "EN→EN", "relevance": 1},
]

def load_resources():
    print("Loading resources...")
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    index = faiss.read_index(str(INDEX_FILE))
    model = SentenceTransformer(MODEL_NAME)
    return metadata, index, model

def calculate_metrics_at_k(results, k_values=[1, 3, 5, 10]):
    """Calculate P@k and R@k"""
    metrics = {}
    for k in k_values:
        correct = sum(1 for r in results if r['rank'] > 0 and r['rank'] <= k)
        precision = correct / len(results)
        recall = correct / len(results)  # Same as precision for single relevant doc
        metrics[f'P@{k}'] = precision * 100
        metrics[f'R@{k}'] = recall * 100
    return metrics

def evaluate_method(queries, metadata, search_func, method_name):
    """Generic evaluation for any search method"""
    results = []
    latencies = []
    
    for item in queries:
        query = item['query']
        expected = item['expected_partial']
        
        start = time.time()
        top_results = search_func(query, k=10)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        
        # Find rank of expected result
        rank = 0
        for i, (vid_name, score) in enumerate(top_results, 1):
            if expected in vid_name:
                rank = i
                break
        
        results.append({
            'query': query,
            'expected': expected,
            'found': rank > 0,
            'rank': rank,
            'top1': top_results[0][0] if top_results else "None",
            'latency_ms': latency,
            'lang_pair': item['lang']
        })
    
    metrics = calculate_metrics_at_k(results)
    metrics['avg_latency'] = np.mean(latencies)
    metrics['p95_latency'] = np.percentile(latencies, 95)
    
    return results, metrics, latencies

# Search functions
def semantic_search(query, k=10):
    metadata, index, model = load_resources()
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)
    return [(metadata[idx]['video_name'], float(1 - distances[0][i])) 
            for i, idx in enumerate(indices[0]) if idx < len(metadata)]

def tfidf_search(query, k=10):
    metadata, _, _ = load_resources()
    corpus = [m['text'] for m in metadata]
    video_names = [m['video_name'] for m in metadata]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]
    
    return [(video_names[i], float(similarities[i])) for i in top_k_indices]

def generate_all_visualizations(semantic_results, tfidf_results, sem_metrics, tfidf_metrics, sem_latencies, tfidf_latencies):
    """Generate all visualization graphs"""
    
    # 1. Accuracy Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['Semantic Search', 'TF-IDF']
    p1_vals = [sem_metrics['P@1'], tfidf_metrics['P@1']]
    p5_vals = [sem_metrics['P@5'], tfidf_metrics['P@5']]
    
    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width/2, p1_vals, width, label='P@1 (%)', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, p5_vals, width, label='P@5 (%)', color='#2ecc71', edgecolor='black')
    
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Video Search: Precision Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'video_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: video_accuracy_comparison.png")
    
    # 2. Latency Box Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [sem_latencies, tfidf_latencies]
    bp = ax.boxplot(data, labels=methods, patch_artist=True)
    
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Query Latency (ms)', fontweight='bold')
    ax.set_title('Video Search: Latency Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'video_latency_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: video_latency_boxplot.png")
    
    # 3. Cross-Language Performance Heatmap
    lang_pairs = {'EN→EN': [], 'EN→HI': [], 'HI→HI': []}
    for r in semantic_results:
        if r['lang_pair'] in lang_pairs:
            lang_pairs[r['lang_pair']].append(1 if r['found'] else 0)
    
    heatmap_data = []
    labels = []
    for pair, results in lang_pairs.items():
        if results:
            acc = np.mean(results) * 100
            heatmap_data.append(acc)
            labels.append(pair)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow([heatmap_data], cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks([0])
    ax.set_yticklabels(['Accuracy'])
    
    for i, val in enumerate(heatmap_data):
        ax.text(i, 0, f'{val:.1f}%', ha='center', va='center', fontweight='bold', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', fontweight='bold')
    ax.set_title('Cross-Language Search Performance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'video_cross_language_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: video_cross_language_heatmap.png")
    
    # 4. Per-k Accuracy Line Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    k_values = [1, 3, 5, 10]
    sem_p_at_k = [sem_metrics[f'P@{k}'] for k in k_values]
    tfidf_p_at_k = [tfidf_metrics[f'P@{k}'] for k in k_values]
    
    ax.plot(k_values, sem_p_at_k, marker='o', linewidth=2, label='Semantic Search', color='#3498db')
    ax.plot(k_values, tfidf_p_at_k, marker='s', linewidth=2, label='TF-IDF', color='#e74c3c')
    
    ax.set_xlabel('k (Top-k Results)', fontweight='bold')
    ax.set_ylabel('Precision@k (%)', fontweight='bold')
    ax.set_title('Precision@k Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'video_precision_at_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: video_precision_at_k.png")

def export_csv_tables(semantic_results, tfidf_results, sem_metrics, tfidf_metrics):
    """Export CSV tables for paper"""
    
    # Per-query results
    with open(RESULTS_DIR / 'video_per_query_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Query', 'Expected Video', 'Semantic Rank', 'TF-IDF Rank', 'Language Pair'])
        for sem_r, tfidf_r in zip(semantic_results, tfidf_results):
            writer.writerow([
                sem_r['query'][:50],
                sem_r['expected'][:40],
                sem_r['rank'] if sem_r['rank'] > 0 else 'Not Found',
                tfidf_r['rank'] if tfidf_r['rank'] > 0 else 'Not Found',
                sem_r['lang_pair']
            ])
    print("✓ Exported: video_per_query_results.csv")
    
    # Method comparison table
    with open(RESULTS_DIR / 'video_method_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Semantic Search', 'TF-IDF'])
        for k in [1, 3, 5, 10]:
            writer.writerow([f'P@{k} (%)', f"{sem_metrics[f'P@{k}']:.1f}", f"{tfidf_metrics[f'P@{k}']:.1f}"])
        writer.writerow(['Avg Latency (ms)', f"{sem_metrics['avg_latency']:.1f}", f"{tfidf_metrics['avg_latency']:.1f}"])
    print("✓ Exported: video_method_comparison.csv")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPREHENSIVE VIDEO SEARCH EVALUATION")
    print("="*60)
    
    # Evaluate both methods
    print("\n[1/4] Evaluating Semantic Search...")
    sem_results, sem_metrics, sem_latencies = evaluate_method(GROUND_TRUTH, None, semantic_search, "Semantic")
    
    print("\n[2/4] Evaluating TF-IDF Baseline...")
    tfidf_results, tfidf_metrics, tfidf_latencies = evaluate_method(GROUND_TRUTH, None, tfidf_search, "TF-IDF")
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<20} | {'Semantic':<12} | {'TF-IDF':<12}")
    print("-" * 50)
    for k in [1, 3, 5, 10]:
        print(f"{'P@' + str(k) + ' (%)':<20} | {sem_metrics[f'P@{k}']:<12.1f} | {tfidf_metrics[f'P@{k}']:<12.1f}")
    print(f"{'Avg Latency (ms)':<20} | {sem_metrics['avg_latency']:<12.1f} | {tfidf_metrics['avg_latency']:<12.1f}")
    
    # Generate visualizations
    print("\n[3/4] Generating visualizations...")
    generate_all_visualizations(sem_results, tfidf_results, sem_metrics, tfidf_metrics, sem_latencies, tfidf_latencies)
    
    # Export tables
    print("\n[4/4] Exporting CSV tables...")
    export_csv_tables(sem_results, tfidf_results, sem_metrics, tfidf_metrics)
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print("="*60)
