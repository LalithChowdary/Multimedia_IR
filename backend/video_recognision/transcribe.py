import os
import whisper
from moviepy.editor import VideoFileClip
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---

# Get the absolute path to the project's backend directory
# This makes the script runnable from anywhere, not just the project root.
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Define the directory where your video files are stored
VIDEO_DIR = BACKEND_DIR / "videos"

# Define the directory where the transcriptions will be saved
TRANSCRIPT_DIR = BACKEND_DIR / "video_recognision" / "transcripts"

# Define the directory for temporary audio files
TEMP_AUDIO_DIR = BACKEND_DIR / "video_recognision" / "temp_audio"

# Choose the Whisper model size.
# Options: "tiny", "base", "small", "medium", "large"
# "small" provides better accuracy for Hindi/multilingual transcription
MODEL_NAME = "small"


def format_timestamp(seconds: float) -> str:
    """Converts a time in seconds to a human-readable HH:MM:SS.mmm format."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def transcribe_videos():
    """
    Extracts audio from videos, transcribes it using the local Whisper model,
    and saves the transcriptions with timestamps.
    """
    print(f"Loading Whisper model: '{MODEL_NAME}'...")
    # This loads the specified Whisper model. The first time a model is used,
    # it will be downloaded automatically and run locally from then on.
    model = whisper.load_model(MODEL_NAME)
    print("Model loaded successfully.")

    # --- Directory Setup ---
    # Ensure the output and temporary directories exist.
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    TEMP_AUDIO_DIR.mkdir(exist_ok=True)
    print(f"Videos will be read from: {VIDEO_DIR}")
    print(f"Transcripts will be saved to: {TRANSCRIPT_DIR}")

    # --- Video Processing ---
    # Get a list of all video files in the directory.
    video_files = [
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm'))
    ]

    if not video_files:
        print(f"No video files found in {VIDEO_DIR}. Please add videos to transcribe.")
        return

    print(f"Found {len(video_files)} video(s) to process.")

    # Process each video file with a progress bar.
    for video_filename in tqdm(video_files, desc="Transcribing Videos"):
        video_path = VIDEO_DIR / video_filename
        video_name = Path(video_filename).stem
        transcript_path = TRANSCRIPT_DIR / f"{video_name}.txt"
        temp_audio_path = TEMP_AUDIO_DIR / f"{video_name}.mp3"

        # --- Skip if Already Processed ---
        if transcript_path.exists():
            print(f"Skipping '{video_filename}', transcript already exists.")
            continue

        try:
            # --- Audio Extraction ---
            print(f"\nProcessing '{video_filename}'...")
            video_clip = VideoFileClip(str(video_path))
            print("Extracting audio...")
            video_clip.audio.write_audiofile(str(temp_audio_path), codec='mp3', logger=None)
            video_clip.close()

            # --- Transcription ---
            print("Transcribing audio with Whisper...")
            # The result object contains detailed segments with timestamps.
            # Force Hindi language to get Devanagari script (not Urdu)
            result = model.transcribe(str(temp_audio_path), fp16=False, language="hi")

            # --- Save Transcription with Timestamps ---
            print(f"Saving transcript with timestamps to '{transcript_path}'...")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                # The result['segments'] is a list of dictionaries,
                # each containing the start time, end time, and text of a segment.
                for segment in result['segments']:
                    start_time = segment['start']
                    end_time = segment['end']
                    text = segment['text']

                    # Format the timestamps into a human-readable format
                    start_formatted = format_timestamp(start_time)
                    end_formatted = format_timestamp(end_time)

                    # Write the formatted line to the file
                    f.write(f"[{start_formatted} --> {end_formatted}] {text.strip()}\n")

            print(f"Successfully transcribed '{video_filename}'.")

        except Exception as e:
            print(f"An error occurred while processing '{video_filename}': {e}")

        finally:
            # --- Cleanup ---
            # Clean up the temporary audio file after processing.
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    print("\nAll videos have been processed.")
    # Optional: Clean up the temp audio directory if it's empty
    if not os.listdir(TEMP_AUDIO_DIR):
        os.rmdir(TEMP_AUDIO_DIR)


if __name__ == "__main__":
    # This allows the script to be run directly from the command line.
    # Example: python backend/video_recognision/transcribe.py
    transcribe_videos()
