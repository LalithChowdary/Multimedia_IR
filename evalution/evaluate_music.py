"""
IMPROVED Music Recognition Evaluation
- Multiple samples per duration for accurate averages
- More robust clip selection (avoid silent sections)
- Better SNR testing
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import time
import csv
import numpy as np
import librosa
from pathlib import Path
from collections import defaultdict
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../backend/music_recognition'))
import fingerprint
from database import FingerprintDB

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent / "backend"
AUDIO_DIR = BACKEND_DIR / "audio_files"
DB_FILE = BACKEND_DIR / "database" / "fingerprints.db"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CLIP_DURATIONS = [5, 10, 15]
SNR_LEVELS = [10, 20, 30, 40, 50]
NOISE_SEED = 42
NUM_SAMPLES_PER_DURATION = 5  # Take 5 samples to get good average

def load_database():
    print("Loading database...")
    db = FingerprintDB()
    db.load()
    return db

def add_noise(audio: np.ndarray, snr_db: float, seed=None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    signal_power = np.mean(audio ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    return audio + noise

def identify_song(audio_sample: np.ndarray, db: FingerprintDB) -> Tuple[str, float]:
    peaks = fingerprint._extract_constellation_map(
        fingerprint._generate_spectrogram(
            fingerprint._preprocess_audio(audio_sample)
        )
    )
    hashes = fingerprint._generate_hashes_from_peaks(peaks, "query")
    matches = db.get_matches(hashes, threshold=3)
    
    if matches:
        best_song_full_path, score, offset = matches[0]
        best_song = Path(best_song_full_path).name
        return best_song, score
    else:
        return None, 0

def select_robust_clip(y, sr, duration_sec):
    """Select clip from middle-to-high energy section to avoid silent parts"""
    duration_samples = int(duration_sec * sr)
    
    if len(y) < duration_samples:
        return y
    
    # Calculate RMS energy in overlapping windows
    window_size = duration_samples
    hop_size = window_size // 2
    energies = []
    positions = []
    
    for i in range(0, len(y) - window_size, hop_size):
        window = y[i:i + window_size]
        energy = np.sqrt(np.mean(window ** 2))
        energies.append(energy)
        positions.append(i)
    
    if not energies:
        # Fallback to center of song
        start = (len(y) - duration_samples) // 2
        return y[start:start + duration_samples]
    
    # Select from top 50% energy sections
    sorted_indices = np.argsort(energies)
    top_half = sorted_indices[len(sorted_indices)//2:]
    selected_idx = random.choice(top_half)
    
    start = positions[selected_idx]
    return y[start:start + duration_samples]

def evaluate_snr_robustness(db, audio_files):
    print("\n[TEST 1/3] SNR Robustness Testing (Multiple Samples)")
    print("-" * 60)
    
    results = {snr: [] for snr in SNR_LEVELS}
    results['clean'] = []
    
    for file_path in audio_files:
        song_name = file_path.name
        print(f"\nTesting: {song_name[:40]}")
        
        try:
            y, sr = librosa.load(str(file_path), sr=fingerprint.SAMPLE_RATE)
        except Exception as e:
            print(f"  Error: {e}")
            continue
        
        if len(y) < 10 * sr:
            print("  Skipping (too short)")
            continue
        
        # Test 3 different clips to get average
        for trial in range(3):
            clip = select_robust_clip(y, sr, 10)
            
            # Clean
            pred, score = identify_song(clip, db)
            is_correct = (pred == song_name)
            results['clean'].append(is_correct)
            
            # Each SNR level
            for snr in SNR_LEVELS:
                noisy_clip = add_noise(clip, snr, seed=NOISE_SEED + trial)
                pred, score = identify_song(noisy_clip, db)
                is_correct = (pred == song_name)
                results[snr].append(is_correct)
        
        clean_acc = np.mean([r for r in results['clean'][-3:]]) * 100
        print(f"  Clean: {clean_acc:.0f}% (3 trials)")
    
    accuracies = {}
    accuracies['Clean'] = np.mean(results['clean']) * 100 if results['clean'] else 0
    for snr in SNR_LEVELS:
        accuracies[snr] = np.mean(results[snr]) * 100 if results[snr] else 0
    
    return accuracies

def evaluate_clip_duration(db, audio_files):
    print("\n[TEST 2/3] Clip Duration Testing (Multiple Samples)")
    print("-" * 60)
    
    results = {dur: [] for dur in CLIP_DURATIONS}
    
    for file_path in audio_files:
        song_name = file_path.name
        print(f"\nTesting: {song_name[:40]}")
        
        try:
            y, sr = librosa.load(str(file_path), sr=fingerprint.SAMPLE_RATE)
        except:
            continue
        
        for duration in CLIP_DURATIONS:
            if len(y) < duration * sr:
                print(f"  {duration}s: Skipped")
                continue
            
            # Test NUM_SAMPLES_PER_DURATION clips
            for trial in range(NUM_SAMPLES_PER_DURATION):
                clip = select_robust_clip(y, sr, duration)
                
                pred, score = identify_song(clip, db)
                is_correct = (pred == song_name)
                results[duration].append(is_correct)
            
            dur_acc = np.mean(results[duration][-NUM_SAMPLES_PER_DURATION:]) * 100
            print(f"  {duration}s: {dur_acc:.0f}% ({NUM_SAMPLES_PER_DURATION} trials)")
    
    accuracies = {dur: np.mean(results[dur]) * 100 if results[dur] else 0 
                  for dur in CLIP_DURATIONS}
    return accuracies

def evaluate_confusion_matrix(db, audio_files):
    print("\n[TEST 3/3] Confusion Matrix (3 trials per song)")
    print("-" * 60)
    
    n_songs = len(audio_files)
    confusion = np.zeros((n_songs, n_songs))
    song_names = [f.name for f in audio_files]
    
    for i, file_path in enumerate(audio_files):
        song_name = file_path.name
        
        try:
            y, sr = librosa.load(str(file_path), sr=fingerprint.SAMPLE_RATE)
        except:
            continue
        
        if len(y) < 10 * sr:
            continue
        
        for trial in range(3):
            clip = select_robust_clip(y, sr, 10)
            pred, score = identify_song(clip, db)
            
            if pred in song_names:
                j = song_names.index(pred)
                confusion[i, j] += 1
        
        print(f"  {song_name[:30]}: {int(confusion[i, i])}/3 correct")
    
    return confusion, song_names

def generate_visualizations(snr_acc, duration_acc, confusion, song_names):
    # 1. SNR Robustness
    fig, ax = plt.subplots(figsize=(10, 6))
    snr_vals = ['Clean'] + SNR_LEVELS
    acc_vals = [snr_acc['Clean']] + [snr_acc[snr] for snr in SNR_LEVELS]
    
    ax.plot(range(len(snr_vals)), acc_vals, marker='o', linewidth=3, markersize=10, color='#9b59b6')
    ax.set_xticks(range(len(snr_vals)))
    ax.set_xticklabels(snr_vals)
    ax.set_xlabel('Noise Level (SNR in dB)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Identification Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Music Recognition: Robustness to Additive Noise\n(Higher SNR = Less Noise)', 
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 110)
    
    for i, val in enumerate(acc_vals):
        ax.text(i, val + 2, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'music_snr_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: music_snr_robustness.png")
    
    # 2. Clip Duration (Should show increasing trend!)
    fig, ax = plt.subplots(figsize=(8, 6))
    durations = list(duration_acc.keys())
    accuracies = list(duration_acc.values())
    
    bars = ax.bar(durations, accuracies, color='#e67e22', edgecolor='black', alpha=0.8, width=2)
    ax.set_xlabel('Clip Duration (seconds)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Identification Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Music Recognition: Clip Duration vs Accuracy\n(Longer clips = More fingerprints = Better matching)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(durations)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'music_duration_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: music_duration_accuracy.png")
    
    # 3. Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    confusion_norm = confusion / confusion.sum(axis=1, keepdims=True).clip(min=1)
    
    im = ax.imshow(confusion_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    short_names = [name[:20] + '...' if len(name) > 20 else name for name in song_names]
    
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Predicted Song', fontweight='bold')
    ax.set_ylabel('Actual Song', fontweight='bold')
    ax.set_title('Music Recognition: Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    for i in range(len(song_names)):
        for j in range(len(song_names)):
            text = ax.text(j, i, f'{confusion_norm[i, j]:.2f}',
                          ha="center", va="center", 
                          color="black" if confusion_norm[i, j] < 0.5 else "white",
                          fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Frequency', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'music_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: music_confusion_matrix.png")
    
    # 4. Performance Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Clean', 'Noisy\n(40dB)', '5s Clip', '10s Clip', '15s Clip']
    values = [snr_acc['Clean'], snr_acc[40], duration_acc[5], duration_acc[10], duration_acc[15]]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Identification Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Music Recognition: Overall Performance Summary', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'music_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: music_performance_summary.png")

def export_csv_tables(snr_acc, duration_acc):
    with open(RESULTS_DIR / 'music_snr_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Condition', 'Accuracy (%)'])
        writer.writerow(['Clean', f"{snr_acc['Clean']:.1f}"])
        for snr in SNR_LEVELS:
            writer.writerow([f'SNR {snr}dB', f"{snr_acc[snr]:.1f}"])
    print("✓ Exported: music_snr_results.csv")
    
    with open(RESULTS_DIR / 'music_duration_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Clip Duration (s)', 'Accuracy (%)'])
        for dur in CLIP_DURATIONS:
            writer.writerow([dur, f"{duration_acc[dur]:.1f}"])
    print("✓ Exported: music_duration_results.csv")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("IMPROVED MUSIC RECOGNITION EVALUATION")
    print("="*60)
    print("\nImprovements:")
    print(f"- {NUM_SAMPLES_PER_DURATION} samples per duration for reliable averages")
    print("- Smart clip selection (avoids silent sections)")
    print("- 3 trials per song for robustness")
    
    db = load_database()
    audio_files = list(AUDIO_DIR.glob("*.mp3")) + list(AUDIO_DIR.glob("*.wav"))
    
    if not audio_files:
        print("❌ No audio files found!")
        exit(1)
    
    print(f"\nFound {len(audio_files)} songs")
    
    snr_acc = evaluate_snr_robustness(db, audio_files)
    duration_acc = evaluate_clip_duration(db, audio_files)
    confusion, song_names = evaluate_confusion_matrix(db, audio_files)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nSNR Robustness:")
    print(f"  Clean:     {snr_acc['Clean']:.1f}%")
    for snr in SNR_LEVELS:
        print(f"  SNR {snr}dB:  {snr_acc[snr]:.1f}%")
    
    print(f"\nClip Duration (SHOULD INCREASE!):")
    for dur in CLIP_DURATIONS:
        print(f"  {dur}s:       {duration_acc[dur]:.1f}%")
    
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    generate_visualizations(snr_acc, duration_acc, confusion, song_names)
    
    print("\n" + "="*60)
    print("Exporting CSV tables...")
    print("="*60)
    export_csv_tables(snr_acc, duration_acc)
    
    print("\n" + "="*60)
    print("✅ MUSIC EVALUATION COMPLETE!")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print("="*60)
