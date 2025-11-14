"""
Debug script for diagnosing real-world audio matching issues.

Usage:
    python debug_matching.py <original_file> <captured_file>
    
This will:
1. Generate fingerprints from both files
2. Compare them in detail
3. Show why matching is failing (if it is)
4. Suggest parameter adjustments
"""

import sys
import numpy as np
from fingerprint import generate_fingerprints, _generate_fingerprints_from_array
from database import FingerprintDB
import librosa

def compare_fingerprints(original_fps, captured_fps):
    """
    Compare two sets of fingerprints in detail.
    """
    print("\n" + "="*60)
    print("FINGERPRINT COMPARISON")
    print("="*60)
    
    # Extract just the hashes
    orig_hashes = set(h for h, _ in original_fps)
    capt_hashes = set(h for h, _ in captured_fps)
    
    # Find overlap
    common_hashes = orig_hashes & capt_hashes
    
    print(f"\nOriginal file hashes: {len(orig_hashes)}")
    print(f"Captured file hashes: {len(capt_hashes)}")
    print(f"Common hashes: {len(common_hashes)}")
    print(f"Match rate: {len(common_hashes) / len(orig_hashes) * 100:.1f}%")
    
    if len(common_hashes) == 0:
        print("\n❌ CRITICAL: NO COMMON HASHES FOUND!")
        print("This means the fingerprints are completely different.")
        print("\nPossible causes:")
        print("1. Audio is too degraded/noisy")
        print("2. Different version of the song")
        print("3. Significant time stretching or pitch shift")
        print("4. Recording is too short")
        print("5. Parameters need adjustment")
        return False
    
    elif len(common_hashes) < len(orig_hashes) * 0.01:
        print("\n⚠️  WARNING: Very few common hashes (<1%)")
        print("Matching might fail. Consider:")
        print("- Increase TARGET_PEAK_DENSITY")
        print("- Increase FAN_VALUE")
        print("- Lower MIN_AMPLITUDE_THRESHOLD")
        return False
    
    else:
        print(f"\n✓ Good hash overlap! ({len(common_hashes)} matches)")
        
        # Analyze time offsets of common hashes
        orig_times = {h: t for h, (_, t) in original_fps}
        capt_times = {h: t for h, (_, t) in captured_fps}
        
        time_diffs = []
        for h in common_hashes:
            if h in orig_times and h in capt_times:
                diff = orig_times[h] - capt_times[h]
                time_diffs.append(diff)
        
        if time_diffs:
            time_diffs = np.array(time_diffs)
            print(f"\nTime offset analysis:")
            print(f"  Mean offset: {np.mean(time_diffs):.1f} frames")
            print(f"  Std deviation: {np.std(time_diffs):.1f} frames")
            print(f"  Min offset: {np.min(time_diffs)}")
            print(f"  Max offset: {np.max(time_diffs)}")
            
            # Check if offsets are consistent
            if np.std(time_diffs) < 50:
                print("  ✓ Offsets are consistent - good for matching!")
            else:
                print("  ⚠️  Offsets are scattered - might indicate timing issues")
        
        return True


def analyze_audio(file_path, label="Audio"):
    """
    Analyze audio file properties.
    """
    print(f"\n{label} Analysis: {file_path}")
    print("-" * 60)
    
    y, sr = librosa.load(file_path, sr=None, mono=True)
    
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(y) / sr:.2f} seconds")
    print(f"Samples: {len(y)}")
    print(f"Max amplitude: {np.max(np.abs(y)):.4f}")
    print(f"RMS energy: {np.sqrt(np.mean(y**2)):.4f}")
    
    # Check for silence
    silence_threshold = 0.01
    silent_samples = np.sum(np.abs(y) < silence_threshold)
    silence_pct = silent_samples / len(y) * 100
    print(f"Silent samples: {silence_pct:.1f}%")
    
    if silence_pct > 50:
        print("⚠️  WARNING: More than 50% silence detected!")
    
    # Check dynamic range
    if np.max(np.abs(y)) < 0.1:
        print("⚠️  WARNING: Very quiet audio (max amplitude < 0.1)")
        print("   Consider increasing recording volume")
    
    return y, sr


def test_matching(original_file, captured_file):
    """
    Full end-to-end matching test.
    """
    print("\n" + "="*60)
    print("REAL-WORLD MATCHING TEST")
    print("="*60)
    
    # Analyze both audio files
    orig_audio, orig_sr = analyze_audio(original_file, "ORIGINAL")
    capt_audio, capt_sr = analyze_audio(captured_file, "CAPTURED")
    
    # Generate fingerprints
    print("\n" + "="*60)
    print("GENERATING FINGERPRINTS")
    print("="*60)
    
    print("\n1. Original file fingerprinting...")
    orig_fps = generate_fingerprints(original_file)
    
    print("\n2. Captured file fingerprinting...")
    capt_fps = generate_fingerprints(captured_file)
    
    # Compare fingerprints
    has_overlap = compare_fingerprints(orig_fps, capt_fps)
    
    # Test database matching
    print("\n" + "="*60)
    print("DATABASE MATCHING TEST")
    print("="*60)
    
    db = FingerprintDB()
    db.add_song(original_file, orig_fps)
    
    print(f"\nSearching for match...")
    matches = db.get_matches(capt_fps, threshold=3)
    
    if matches:
        print(f"\n✅ SUCCESS! Found {len(matches)} match(es)")
        for i, (song_id, score, offset) in enumerate(matches[:3], 1):
            print(f"\n  #{i}: {song_id}")
            print(f"      Score: {score}")
            print(f"      Offset: {offset} frames ({offset * 64 / 8000:.2f} seconds)")
    else:
        print("\n❌ NO MATCH FOUND")
        print("\nTroubleshooting steps:")
        print("1. Check if fingerprints have any overlap (see above)")
        print("2. Try lowering MIN_MATCH_THRESHOLD to 2 or 1")
        print("3. Increase CLUSTER_TOLERANCE to 10 or 20")
        print("4. Verify audio quality (volume, noise, duration)")
    
    return matches is not None and len(matches) > 0


def simulate_capture(audio_file, noise_level=0.05, volume=0.7):
    """
    Simulate microphone capture by adding noise and volume variation.
    Useful for testing without actual recording equipment.
    """
    print("\n" + "="*60)
    print("SIMULATED CAPTURE TEST")
    print("="*60)
    
    print(f"\nLoading: {audio_file}")
    y, sr = librosa.load(audio_file, sr=8000, mono=True)
    
    # Simulate capture effects
    print(f"Applying capture simulation:")
    print(f"  - Volume: {volume * 100:.0f}%")
    print(f"  - Noise level: {noise_level}")
    
    # Reduce volume
    y_captured = y * volume
    
    # Add white noise
    noise = np.random.randn(len(y)) * noise_level
    y_captured = y_captured + noise
    
    # Clip to [-1, 1]
    y_captured = np.clip(y_captured, -1, 1)
    
    # Generate fingerprints for both
    print("\nOriginal fingerprints...")
    orig_fps = _generate_fingerprints_from_array(y, audio_file)
    
    print("\nCaptured fingerprints...")
    capt_fps = _generate_fingerprints_from_array(y_captured, "captured")
    
    # Compare
    compare_fingerprints(orig_fps, capt_fps)
    
    # Test matching
    db = FingerprintDB()
    db.add_song(audio_file, orig_fps)
    
    print("\nMatching test...")
    matches = db.get_matches(capt_fps, threshold=3)
    
    if matches:
        print(f"\n✅ SIMULATED CAPTURE MATCHED!")
        best = matches[0]
        print(f"   Song: {best[0]}")
        print(f"   Score: {best[1]}")
        print(f"   Offset: {best[2]}")
        return True
    else:
        print(f"\n❌ SIMULATED CAPTURE FAILED TO MATCH")
        return False


def suggest_parameters(original_fps, captured_fps):
    """
    Analyze fingerprints and suggest parameter adjustments.
    """
    print("\n" + "="*60)
    print("PARAMETER RECOMMENDATIONS")
    print("="*60)
    
    orig_hashes = set(h for h, _ in original_fps)
    capt_hashes = set(h for h, _ in captured_fps)
    common = orig_hashes & capt_hashes
    
    overlap_rate = len(common) / len(orig_hashes) if orig_hashes else 0
    
    print(f"\nCurrent overlap rate: {overlap_rate * 100:.1f}%")
    
    if overlap_rate < 0.01:
        print("\nSEVERE: Almost no overlap")
        print("\nRecommended changes:")
        print("1. TARGET_PEAK_DENSITY: Increase to 15-20")
        print("2. FAN_VALUE: Increase to 20")
        print("3. MIN_AMPLITUDE_THRESHOLD: Decrease to 5")
        print("4. FREQ_WINDOW: Increase to 3000")
        print("5. Capture longer audio (15+ seconds)")
        
    elif overlap_rate < 0.05:
        print("\nPOOR: Very low overlap")
        print("\nRecommended changes:")
        print("1. TARGET_PEAK_DENSITY: Increase to 12-15")
        print("2. FAN_VALUE: Increase to 15")
        print("3. MIN_AMPLITUDE_THRESHOLD: Decrease to 8")
        
    elif overlap_rate < 0.20:
        print("\nFAIR: Low overlap, might work with adjustments")
        print("\nRecommended changes:")
        print("1. MIN_MATCH_THRESHOLD: Lower to 2")
        print("2. CLUSTER_TOLERANCE: Increase to 10")
        print("3. Consider increasing capture volume/quality")
        
    else:
        print("\nGOOD: Overlap is sufficient!")
        print("If matching still fails, try:")
        print("1. MIN_MATCH_THRESHOLD: Lower to 2")
        print("2. CLUSTER_TOLERANCE: Increase to 10")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python debug_matching.py <audio_file>                    # Simulate capture")
        print("  python debug_matching.py <original> <captured>           # Compare two files")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Simulation mode
        print("Running in SIMULATION mode")
        audio_file = sys.argv[1]
        
        # Test with different noise levels
        for noise in [0.01, 0.05, 0.1]:
            print(f"\n{'='*60}")
            print(f"Testing with noise level: {noise}")
            print(f"{'='*60}")
            success = simulate_capture(audio_file, noise_level=noise, volume=0.7)
            
            if not success:
                print(f"\n⚠️  Failed at noise level {noise}")
                break
    
    else:
        # Real comparison mode
        print("Running in COMPARISON mode")
        original = sys.argv[1]
        captured = sys.argv[2]
        
        success = test_matching(original, captured)
        
        # Load both for parameter suggestions
        orig_fps = generate_fingerprints(original)
        capt_fps = generate_fingerprints(captured)
        suggest_parameters(orig_fps, capt_fps)
        
        if success:
            print("\n" + "="*60)
            print("✅ MATCHING SUCCESSFUL!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ MATCHING FAILED - See recommendations above")
            print("="*60)