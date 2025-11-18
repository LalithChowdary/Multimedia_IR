import os
import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from scipy import signal
from scipy.signal import butter, sosfilt

# --- Shazam Algorithm Constants (Real-World Optimized) ---

# Audio processing parameters
SAMPLE_RATE = 8000  # 8KHz as per paper

# STFT Parameters - optimized for real-world matching
FFT_WINDOW_SIZE = 2048  # INCREASED for better frequency resolution
FFT_HOP_LENGTH = 64     # INCREASED for more stable peaks

# Peak finding parameters - MORE AGGRESSIVE for real-world
# Paper: "Candidate peaks are chosen according to amplitude"
MIN_AMPLITUDE_THRESHOLD = 10  # dB above noise floor (lowered for captured audio)

# Neighborhood size for peak detection - LARGER for stability
PEAK_NEIGHBORHOOD_TIME = 10   # frames (doubled)
PEAK_NEIGHBORHOOD_FREQ = 10   # frequency bins

# Density control - CRITICAL for real-world matching
# Need MORE peaks to ensure overlap between original and captured
TARGET_PEAK_DENSITY = 20.0  # peaks per second (MUCH higher)

# Fingerprint generation parameters - OPTIMIZED for robustness
# Paper: "fan-out factor" F
FAN_VALUE = 20  # INCREASED for more redundancy in noisy conditions

# Target zone definition - WIDER for matching across distortions
MIN_TIME_DELTA = 0      
MAX_TIME_DELTA = 200    
FREQ_WINDOW = 2000      # DOUBLED - allows more freq variation

# Frequency range to focus on (Hz)
# Human speech is 80-300Hz, music is 80-15000Hz
# Focus on mid-high frequencies where music is distinct
MIN_FREQ_HZ = 300      # Ignore low rumble and bass (often distorted)
MAX_FREQ_HZ = 4000     # Nyquist at 8KHz is 4000Hz


def generate_fingerprints(file_path: str):
    """
    Generates audio fingerprints using the Shazam algorithm.
    OPTIMIZED for real-world audio capture scenarios.
    
    Args:
        file_path: Path to the audio file.
    
    Returns:
        A set of (hash, (song_id, time_offset)) tuples.
    """
    try:
        # Load audio at 8KHz mono (as per Shazam paper)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        song_id = os.path.basename(file_path)
        return _generate_fingerprints_from_array(y, song_id)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return set()


def _generate_fingerprints_from_array(y: np.ndarray, song_id: str):
    """
    Core fingerprinting logic implementing Shazam's constellation map approach.
    ENHANCED for real-world microphone capture.
    
    Process:
    1. Preprocess audio (denoise, normalize, filter)
    2. Generate spectrogram
    3. Extract constellation map (robust peaks)
    4. Create combinatorial hashes using anchor-target pairing
    
    Args:
        y: Audio time series
        song_id: Identifier for the audio
    
    Returns:
        Set of (hash, (song_id, time_offset)) tuples
    """
    # CRITICAL: Preprocess for real-world capture
    y = _preprocess_audio(y)
    
    # Step 1: Create spectrogram using STFT
    spectrogram = _generate_spectrogram(y)
    
    # Step 2: Extract constellation map (robust peaks)
    peaks = _extract_constellation_map(spectrogram)
    
    print(f"Extracted {len(peaks)} constellation peaks")
    
    # Step 3: Generate combinatorial hashes from peaks
    hashes = _generate_hashes_from_peaks(peaks, song_id)
    
    print(f"Generated {len(hashes)} fingerprint hashes")
    
    return hashes


def _preprocess_audio(y: np.ndarray):
    """
    Preprocess audio for robust fingerprinting.
    CRITICAL for matching captured audio to database originals.
    
    Steps:
    1. Normalize amplitude
    2. High-pass filter (remove low-frequency rumble)
    3. Pre-emphasis (boost high frequencies)
    4. Noise gate (remove very quiet sections)
    """
    # 1. Normalize to [-1, 1] range
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    # 2. High-pass filter at 300Hz to remove low-frequency noise
    # Room rumble, AC hum, and bass are often distorted in capture
    sos = butter(4, 300, 'hp', fs=SAMPLE_RATE, output='sos')
    y = sosfilt(sos, y)
    
    # 3. Pre-emphasis filter to boost high frequencies
    # High frequencies contain more distinctive features
    y = librosa.effects.preemphasis(y, coef=0.97)
    
    # 4. Noise gate - zero out very quiet samples
    # Helps remove silent sections and background hiss
    threshold = 0.01
    y[np.abs(y) < threshold] = 0
    
    # 5. Re-normalize after filtering
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    return y


def _generate_spectrogram(y: np.ndarray):
    """
    Generate spectrogram using Short-Time Fourier Transform.
    ENHANCED for real-world matching.
    
    Returns magnitude spectrogram with log scaling for better peak detection.
    """
    # Use Hanning window for better frequency resolution
    window = signal.windows.hann(FFT_WINDOW_SIZE)
    
    # Compute STFT
    stft = librosa.stft(
        y,
        n_fft=FFT_WINDOW_SIZE,
        hop_length=FFT_HOP_LENGTH,
        window=window
    )
    
    # Get magnitude spectrogram
    spectrogram = np.abs(stft)
    
    # Focus on frequency range of interest (300Hz - 4000Hz)
    # This removes unreliable low and very high frequencies
    freq_bins = spectrogram.shape[0]
    freq_resolution = SAMPLE_RATE / FFT_WINDOW_SIZE
    
    min_bin = int(MIN_FREQ_HZ / freq_resolution)
    max_bin = int(MAX_FREQ_HZ / freq_resolution)
    
    # Crop to frequency range
    spectrogram = spectrogram[min_bin:max_bin, :]
    
    # Convert to dB scale with noise floor
    # Add small epsilon to avoid log(0)
    spectrogram = librosa.amplitude_to_db(spectrogram + 1e-10, ref=np.max)
    
    # Apply median filtering to reduce noise spikes
    # This helps with random noise in captured audio
    from scipy.ndimage import median_filter
    spectrogram = median_filter(spectrogram, size=(3, 3))
    
    return spectrogram


def _extract_constellation_map(spectrogram: np.ndarray):
    """
    Extract constellation map: robust spectrogram peaks.
    ENHANCED to find peaks that survive real-world capture.
    
    Key changes:
    - More aggressive peak detection
    - Better density control
    - Amplitude-weighted selection
    
    Returns:
        List of (time_idx, freq_idx) tuples representing peaks
    """
    # Get spectrogram dimensions
    freq_bins, time_frames = spectrogram.shape
    
    # Apply maximum filter to find local maxima
    struct = np.ones((PEAK_NEIGHBORHOOD_FREQ, PEAK_NEIGHBORHOOD_TIME))
    local_max = maximum_filter(spectrogram, footprint=struct) == spectrogram
    
    # Apply amplitude threshold (lower for captured audio)
    threshold = spectrogram.max() - MIN_AMPLITUDE_THRESHOLD
    amplitude_filter = spectrogram > threshold
    
    # Combine filters to get peak candidates
    peaks_mask = local_max & amplitude_filter
    
    # Get peak coordinates and amplitudes
    freq_idx, time_idx = np.where(peaks_mask)
    peak_amplitudes = spectrogram[freq_idx, time_idx]
    
    # Create array of peaks with their properties
    peaks_data = np.column_stack((time_idx, freq_idx, peak_amplitudes))
    
    # Sort by amplitude (descending) to prioritize strongest peaks
    peaks_data = peaks_data[peaks_data[:, 2].argsort()[::-1]]
    
    # Apply density control - INCREASED for real-world
    duration = time_frames * FFT_HOP_LENGTH / SAMPLE_RATE
    target_num_peaks = int(TARGET_PEAK_DENSITY * duration)
    
    # Take top N peaks by amplitude
    if len(peaks_data) > target_num_peaks:
        peaks_data = peaks_data[:target_num_peaks]
    
    # Convert back to list of (time, freq) tuples
    # Adjust freq_idx to account for cropped frequency range
    freq_offset = int(MIN_FREQ_HZ / (SAMPLE_RATE / FFT_WINDOW_SIZE))
    peaks_list = [(int(t), int(f) + freq_offset) for t, f, _ in peaks_data]
    
    return peaks_list


def _generate_hashes_from_peaks(peaks: list, song_id: str):
    """
    Generate combinatorial hashes using anchor-target point pairing.
    ENHANCED for maximum redundancy in real-world scenarios.
    
    Key changes:
    - Higher fan-out for more hash redundancy
    - Wider frequency window for distortion tolerance
    
    Returns:
        Set of (hash, (song_id, anchor_time)) tuples
    """
    hashes = set()
    
    # Sort peaks by time for efficient pairing
    peaks = sorted(peaks, key=lambda x: x[0])
    
    # For each peak, use it as an anchor point
    for i, (t1, f1) in enumerate(peaks):
        # Define target zone boundaries
        target_zone_start = i + 1
        
        # Count targets added for this anchor
        targets_added = 0
        
        # Look for target points in the target zone
        for j in range(target_zone_start, len(peaks)):
            if targets_added >= FAN_VALUE:
                break
            
            t2, f2 = peaks[j]
            
            # Check if target point is within time bounds
            time_delta = t2 - t1
            if time_delta < MIN_TIME_DELTA:
                continue
            if time_delta > MAX_TIME_DELTA:
                break
            
            # Check if target point is within frequency bounds
            freq_delta = abs(f2 - f1)
            if freq_delta > FREQ_WINDOW:
                continue
            
            # Create hash from (freq1, freq2, time_delta)
            hash_value = _create_hash(f1, f2, time_delta)
            
            # Store hash with song_id and anchor time
            hashes.add((hash_value, (song_id, t1)))
            targets_added += 1
    
    return hashes


def _create_hash(f1: int, f2: int, t_delta: int):
    """
    Create a deterministic hash from frequency pair and time delta.
    
    Uses bit-packing for deterministic 32-bit hash:
    - 10 bits for f1 (0-1023)
    - 10 bits for f2 (0-1023)  
    - 12 bits for t_delta (0-4095)
    """
    # Clamp values to fit in allocated bits
    f1_clamped = min(int(f1), 1023)
    f2_clamped = min(int(f2), 1023)
    t_delta_clamped = min(int(t_delta), 4095)
    
    # Pack into 32-bit integer
    hash_value = (f1_clamped << 22) | (f2_clamped << 12) | t_delta_clamped
    
    return hash_value


if __name__ == '__main__':
    # Test the fingerprinting algorithm
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Testing fingerprint generation on: {test_file}")
        
        fingerprints = generate_fingerprints(test_file)
        print(f"Generated {len(fingerprints)} fingerprints")
        
        # Show first few fingerprints
        for i, (hash_val, (song_id, time_offset)) in enumerate(list(fingerprints)[:10]):
            print(f"  Hash: {hash_val:08x}, Time: {time_offset}, Song: {song_id}")
    else:
        print("Usage: python fingerprint.py <audio_file>")
        print("\nThis version is optimized for real-world audio capture!")