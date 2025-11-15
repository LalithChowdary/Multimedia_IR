"""
Audio extraction from video files using moviepy.
Extracts audio track and saves as WAV for transcription.
"""

import os
from pathlib import Path
from typing import Optional
from moviepy.editor import VideoFileClip
import logging

logger = logging.getLogger(__name__)

# Base directory for videos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(BASE_DIR, 'videos')
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, 'temp', 'audio')

# Create temp directory if it doesn't exist
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)


def extract_audio_from_video(video_path: str, output_audio_path: Optional[str] = None) -> str:
    """
    Extract audio from video file and save as WAV.
    
    Args:
        video_path: Path to the input video file
        output_audio_path: Optional path for output audio file.
                          If None, saves to temp directory with same name.
    
    Returns:
        Path to the extracted audio file
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        Exception: If audio extraction fails
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Generate output path if not provided
    if output_audio_path is None:
        video_name = Path(video_path).stem
        output_audio_path = os.path.join(TEMP_AUDIO_DIR, f"{video_name}.wav")
    
    try:
        logger.info(f"Extracting audio from: {video_path}")
        
        # Load video and extract audio
        video = VideoFileClip(video_path)
        
        # Check if video has audio
        if video.audio is None:
            raise ValueError(f"Video has no audio track: {video_path}")
        
        # Extract audio and save as WAV
        # Using 16kHz sample rate for Whisper (optimal for speech)
        video.audio.write_audiofile(
            output_audio_path,
            fps=16000,  # 16kHz sample rate (Whisper recommendation)
            nbytes=2,   # 16-bit audio
            codec='pcm_s16le',  # PCM format
            verbose=False,
            logger=None
        )
        
        # Close video to free resources
        video.close()
        
        logger.info(f"Audio extracted successfully: {output_audio_path}")
        return output_audio_path
        
    except Exception as e:
        logger.error(f"Failed to extract audio from {video_path}: {e}")
        raise


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video in seconds.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds
    """
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()
        return duration
    except Exception as e:
        logger.error(f"Failed to get video duration: {e}")
        raise


def get_video_info(video_path: str) -> dict:
    """
    Get metadata about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video metadata
    """
    try:
        video = VideoFileClip(video_path)
        
        info = {
            'duration': video.duration,
            'fps': video.fps,
            'size': video.size,  # (width, height)
            'has_audio': video.audio is not None,
            'filename': os.path.basename(video_path),
            'path': video_path
        }
        
        video.close()
        return info
        
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        raise


def cleanup_temp_audio(audio_path: str):
    """
    Remove temporary audio file.
    
    Args:
        audio_path: Path to the audio file to remove
    """
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info(f"Cleaned up temporary audio: {audio_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup audio file {audio_path}: {e}")


if __name__ == '__main__':
    # Test audio extraction
    import sys
    
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        
        print(f"Extracting audio from: {video_file}")
        audio_path = extract_audio_from_video(video_file)
        print(f"Audio saved to: {audio_path}")
        
        info = get_video_info(video_file)
        print(f"\nVideo Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("Usage: python audio_extractor.py <video_file>")
