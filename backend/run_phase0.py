import os
import glob
from backend.app.fingerprint import generate_fingerprints
from backend.app.database import FingerprintDB

# --- Constants ---
AUDIO_DIR = 'backend/audio_files'

def ingest_songs():
    """
    Processes all audio files in the AUDIO_DIR, generates their fingerprints,
    and stores them in the database.
    """
    db = FingerprintDB()
    db.load()  # Load existing database to avoid re-processing

    # Find all audio files in the directory
    # We'll support a few common formats.
    audio_files = glob.glob(os.path.join(AUDIO_DIR, '*.mp3'))
    audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, '*.wav')))
    
    for file_path in audio_files:
        song_id = os.path.basename(file_path)
        
        # A simple check to see if the song is already in the DB.
        # This is not foolproof but good for a simple script.
        # A more robust system would check hashes.
        if any(song_id in s for s in db.db.values()):
             print(f"'{song_id}' already in database. Skipping.")
             continue

        print(f"Processing '{song_id}'...")
        fingerprints = generate_fingerprints(file_path)
        
        if fingerprints:
            db.add_song(song_id, fingerprints)
            print(f"  Added {len(fingerprints)} fingerprints to the database.")
        else:
            print(f"  Could not generate fingerprints for '{song_id}'.")

    db.save()
    print("\nDatabase saved.")

def recognize_song(sample_path: str):
    """
    Recognizes a sample audio clip by fingerprinting it and matching it
    against the database.
    """
    if not os.path.exists(sample_path):
        print(f"Sample file not found: {sample_path}")
        return

    db = FingerprintDB()
    db.load()

    print(f"\nRecognizing sample: '{os.path.basename(sample_path)}'...")
    
    sample_fingerprints = generate_fingerprints(sample_path)
    
    if not sample_fingerprints:
        print("Could not generate fingerprints for the sample.")
        return

    matches = db.get_matches(sample_fingerprints)

    if matches:
        best_match = matches[0]
        print(f"\n--- Match Found ---")
        print(f"Song: {best_match[0]}")
        print(f"Confidence: {best_match[1]}")
        print(f"-------------------")
    else:
        print("\nNo match found.")


if __name__ == '__main__':
    # --- Main Execution ---
    
    # 1. Ingest songs into the database
    # This will process all .mp3 and .wav files in the audio_files directory.
    # Make sure you've placed your audio files there.
    print("--- Starting Song Ingestion ---")
    ingest_songs()
    print("-------------------------------\n")

    # 2. Recognize a sample
    # To test this, you should create a short clip (5-10 seconds) from one
    # of the songs you ingested. You can use an audio editor like Audacity.
    # Or, you can just use one of the original files as a "perfect" sample.
    
    # Replace this with the path to your sample clip
    # For example: sample_to_recognize = 'backend/audio_files/your_song_clip.wav'
    
    # Find the first song in the audio_files directory and use it as a sample
    audio_files = glob.glob(os.path.join(AUDIO_DIR, '*.mp3'))
    audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, '*.wav')))
    
    if audio_files:
        sample_to_recognize = audio_files[0]
        recognize_song(sample_to_recognize)
    else:
        print("No audio files found in 'backend/audio_files' to use as a sample.")
        print("Please add some audio files and run the script again.")
