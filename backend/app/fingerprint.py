import librosa
import numpy as np
import os
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, iterate_structure

# --- Constants ---
# Downsampling rate. We don't need full quality for fingerprinting.
# 22050 Hz is a common choice.
SAMPLE_RATE = 22050

# FFT window size. Determines the frequency resolution.
# 4096 is a good starting point.
FFT_WINDOW_SIZE = 4096

# Hop length. Number of samples between successive FFT windows.
# A smaller hop length gives more temporal resolution.
FFT_HOP_LENGTH = 2048

# --- Peak Finding Parameters ---
# This defines a square neighborhood of 20x20 around a peak.
# A peak is only kept if it's the maximum in this neighborhood.
PEAK_NEIGHBORHOOD_SIZE = 20

# --- Fingerprinting Parameters ---
# The "target zone" for pairing peaks.
# We'll pair a peak with other peaks that appear after it in time
# and are within a certain frequency and time range.
TARGET_ZONE_T_START = 0.1  # Start of target zone (seconds)
TARGET_ZONE_T_END = 1.0    # End of target zone (seconds)
TARGET_ZONE_F_BINS = 100   # Frequency range (in FFT bins)


def generate_fingerprints(file_path: str):
    """
    Generates a set of audio fingerprints for a given audio file.

    This function now acts as a wrapper that loads the audio file
    and passes the data to the core fingerprinting logic.

    Args:
        file_path: The path to the audio file.

    Returns:
        A set of hashes for the given audio data.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        song_id = os.path.basename(file_path)
        return _generate_fingerprints_from_array(y, song_id)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return set()

def _generate_fingerprints_from_array(y: np.ndarray, song_id: str):
    """
    Generates a set of audio fingerprints from a raw audio array.

    This process involves:
    1. Creating a spectrogram from the audio data.
    2. Identifying peaks (local maxima) in the spectrogram.
    3. Creating combinatorial hashes from pairs of peaks.

    Args:
        y: The audio time series as a NumPy array.
        song_id: The identifier for the song/stream.

    Returns:
        A set of hashes, where each hash is a tuple:
        (hash_value, (song_id, anchor_time_offset))
    """
    # 1. Create spectrogram
    spectrogram = np.abs(librosa.stft(y,
                                      n_fft=FFT_WINDOW_SIZE,
                                      hop_length=FFT_HOP_LENGTH))

    # 2. Find peaks
    peaks = _find_spectrogram_peaks(spectrogram)

    # 3. Create hashes
    hashes = _create_hashes(peaks, song_id)

    return hashes


def _find_spectrogram_peaks(spectrogram: np.ndarray):
    """
    Finds local maxima (peaks) in a spectrogram.
    """
    # We use a maximum filter to find peaks. A point is a peak if it's the
    # maximum value in a neighborhood of a certain size.
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram
    
    # The local_max is a boolean array. We want the coordinates of the peaks.
    # We also apply a threshold to discard very quiet peaks.
    # This threshold can be tuned.
    detected_peaks = (spectrogram > np.median(spectrogram) * 1.5) & local_max
    
    # Get the (frequency, time) coordinates of the peaks
    freq_bins, time_indices = np.where(detected_peaks)
    
    # Return a list of (time, frequency) pairs
    return list(zip(time_indices, freq_bins))


def _create_hashes(peaks: list, song_id: str):
    """
    Creates combinatorial hashes from a list of spectrogram peaks.
    """
    hashes = set()
    
    # For each peak, we create a hash with other peaks in its "target zone".
    for i, (t1, f1) in enumerate(peaks):
        # Define the target zone for the current peak (t1, f1)
        t_min = t1 + librosa.time_to_frames(TARGET_ZONE_T_START, sr=SAMPLE_RATE, hop_length=FFT_HOP_LENGTH)
        t_max = t1 + librosa.time_to_frames(TARGET_ZONE_T_END, sr=SAMPLE_RATE, hop_length=FFT_HOP_LENGTH)
        f_min = f1 - TARGET_ZONE_F_BINS
        f_max = f1 + TARGET_ZONE_F_BINS

        # Iterate through subsequent peaks to find pairs
        for (t2, f2) in peaks[i+1:]:
            if t_min <= t2 <= t_max and f_min <= f2 <= f_max:
                # This is the "secret sauce". The hash is a combination of the
                # two peak frequencies and the time difference between them.
                # This makes the hash robust to time shifts.
                time_delta = t2 - t1
                hash_val = hash((f1, f2, time_delta))
                
                # We store the hash along with the song ID and the time of the
                # first peak. This is crucial for matching later.
                hashes.add((hash_val, (song_id, t1)))

    return hashes

if __name__ == '__main__':
    # This is a simple test to run the script directly.
    # Replace with a path to an actual audio file.
    # Make sure you have a file in 'backend/audio_files/'
    
    # Example usage:
    # test_file = 'backend/audio_files/your_song.mp3'
    # fingerprints = generate_fingerprints(test_file)
    # print(f"Generated {len(fingerprints)} fingerprints for {test_file}")
    # print("Example fingerprint:", list(fingerprints)[0] if fingerprints else "None")
    pass
