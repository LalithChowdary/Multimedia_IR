import os
import pickle
import random
import time
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Import local modules
import fingerprint
from database import FingerprintDB

# --- Configuration ---
BACKEND_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BACKEND_DIR / "audio_files"
DB_FILE = BACKEND_DIR / "database" / "fingerprints.db"

TEST_DURATION = 10  # seconds
NOISE_LEVEL = 0.005  # Amplitude of white noise

def load_database():
    print(f"Loading database from {DB_FILE}...")
    db = FingerprintDB()
    db.load()
    return db

def add_noise(audio: np.ndarray, noise_level: float) -> np.ndarray:
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

def identify_song(audio_sample: np.ndarray, db: FingerprintDB, song_name: str) -> Tuple[str, float]:
    # 1. Generate Fingerprints for the sample
    # The fingerprint format is: (hash, (song_id, time_offset))
    peaks = fingerprint._extract_constellation_map(
        fingerprint._generate_spectrogram(
            fingerprint._preprocess_audio(audio_sample)
        )
    )
    hashes = fingerprint._generate_hashes_from_peaks(peaks, "query")
    
    # 2. Use the database's get_matches method
    matches = db.get_matches(hashes, threshold=3)
    
    # 3. Return the top match
    if matches:
        best_song_full_path, score, offset = matches[0]
        # Extract filename from full path
        best_song = Path(best_song_full_path).name
        return best_song, score
    else:
        return None, 0

def run_evaluation():
    db = load_database()
    audio_files = list(AUDIO_DIR.glob("*.mp3")) + list(AUDIO_DIR.glob("*.wav"))
    
    if not audio_files:
        print("No audio files found for testing!")
        return

    print(f"\n--- Starting Music Recognition Evaluation ---")
    print(f"Test Duration: {TEST_DURATION}s")
    print(f"Noise Level:   {NOISE_LEVEL}")
    print(f"Total Songs:   {len(audio_files)}")
    
    results = []
    
    for file_path in audio_files:
        song_name = file_path.name
        print(f"\nTesting: {song_name}")
        
        # Load Audio
        try:
            y, sr = librosa.load(str(file_path), sr=fingerprint.SAMPLE_RATE)
        except Exception as e:
            print(f"  Error loading file: {e}")
            continue
            
        if len(y) < TEST_DURATION * sr:
            print("  Skipping (too short)")
            continue
            
        # Extract Random Clip
        start_sample = random.randint(0, len(y) - int(TEST_DURATION * sr))
        end_sample = start_sample + int(TEST_DURATION * sr)
        clip_clean = y[start_sample:end_sample]
        
        # Create Noisy Clip
        clip_noisy = add_noise(clip_clean, NOISE_LEVEL)
        
        # --- Test Clean ---
        t0 = time.time()
        pred_clean, score_clean = identify_song(clip_clean, db, song_name)
        t_clean = time.time() - t0
        
        is_correct_clean = (pred_clean == song_name)
        print(f"  [Clean] Pred: {pred_clean} (Score: {score_clean}) | Correct: {is_correct_clean} | Time: {t_clean:.3f}s")
        
        # --- Test Noisy ---
        t0 = time.time()
        pred_noisy, score_noisy = identify_song(clip_noisy, db, song_name)
        t_noisy = time.time() - t0
        
        is_correct_noisy = (pred_noisy == song_name)
        print(f"  [Noisy] Pred: {pred_noisy} (Score: {score_noisy}) | Correct: {is_correct_noisy} | Time: {t_noisy:.3f}s")
        
        results.append({
            "song": song_name,
            "clean_correct": is_correct_clean,
            "noisy_correct": is_correct_noisy,
            "clean_time": t_clean,
            "noisy_time": t_noisy
        })
        
    # Summary
    total = len(results)
    acc_clean = sum(1 for r in results if r['clean_correct']) / total * 100
    acc_noisy = sum(1 for r in results if r['noisy_correct']) / total * 100
    avg_time = sum(r['clean_time'] for r in results) / total
    
    print("\n" + "="*40)
    print("MUSIC RECOGNITION RESULTS")
    print("="*40)
    print(f"Clean Accuracy: {acc_clean:.1f}%")
    print(f"Noisy Accuracy: {acc_noisy:.1f}%")
    print(f"Avg Time:       {avg_time:.3f} s")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
