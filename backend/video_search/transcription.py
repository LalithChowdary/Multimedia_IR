"""
Video transcription using OpenAI Whisper (local/offline).
Generates timestamped transcripts for semantic search.
"""

import os
import whisper
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a transcript segment with timestamp."""
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    
    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'start_time': self.start_time,
            'end_time': self.end_time
        }


class WhisperTranscriber:
    """
    Handles video transcription using Whisper model.
    Uses local model for offline operation.
    """
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_name: Whisper model size
                - 'tiny': Fastest, least accurate (~1GB RAM)
                - 'base': Good balance (~1GB RAM) - RECOMMENDED
                - 'small': Better accuracy (~2GB RAM)
                - 'medium': High accuracy (~5GB RAM)
                - 'large': Best accuracy (~10GB RAM)
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Initializing Whisper transcriber with model: {model_name}")
    
    def load_model(self):
        """Load Whisper model into memory."""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
    
    def transcribe_audio(
        self, 
        audio_path: str,
        language: Optional[str] = None,
        chunk_length: int = 30
    ) -> List[TranscriptSegment]:
        """
        Transcribe audio file with timestamps.
        
        Args:
            audio_path: Path to audio file (WAV format)
            language: Optional language code (e.g., 'en', 'es')
                     If None, Whisper will auto-detect
            chunk_length: Target length for each segment in seconds
        
        Returns:
            List of TranscriptSegment objects
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load model if not already loaded
        self.load_model()
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            # Transcribe with Whisper
            # word_timestamps=True gives us precise timing for better chunking
            result = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                verbose=False
            )
            
            # Extract segments with timestamps
            segments = []
            
            for segment in result['segments']:
                # Create transcript segment
                transcript_segment = TranscriptSegment(
                    text=segment['text'].strip(),
                    start_time=segment['start'],
                    end_time=segment['end']
                )
                segments.append(transcript_segment)
            
            logger.info(f"Transcription complete. Generated {len(segments)} segments")
            
            # Optionally re-chunk to target length
            if chunk_length > 0:
                segments = self._rechunk_segments(segments, chunk_length)
            
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _rechunk_segments(
        self, 
        segments: List[TranscriptSegment], 
        target_length: int
    ) -> List[TranscriptSegment]:
        """
        Re-chunk segments to approximate target length.
        Combines short segments and splits long ones.
        
        Args:
            segments: Original segments from Whisper
            target_length: Target length in seconds
            
        Returns:
            Re-chunked segments
        """
        if not segments:
            return []
        
        chunked = []
        current_chunk_text = []
        current_start = segments[0].start_time
        current_end = segments[0].end_time
        
        for segment in segments:
            segment_duration = current_end - current_start
            
            # If adding this segment would exceed target, save current chunk
            if segment_duration >= target_length and current_chunk_text:
                chunked.append(TranscriptSegment(
                    text=' '.join(current_chunk_text),
                    start_time=current_start,
                    end_time=current_end
                ))
                current_chunk_text = []
                current_start = segment.start_time
            
            # Add segment to current chunk
            current_chunk_text.append(segment.text)
            current_end = segment.end_time
        
        # Add final chunk
        if current_chunk_text:
            chunked.append(TranscriptSegment(
                text=' '.join(current_chunk_text),
                start_time=current_start,
                end_time=current_end
            ))
        
        logger.info(f"Re-chunked {len(segments)} segments into {len(chunked)} chunks")
        return chunked
    
    def transcribe_with_metadata(
        self,
        audio_path: str,
        video_name: str,
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe and return with metadata for database storage.
        
        Args:
            audio_path: Path to audio file
            video_name: Name of the source video
            language: Optional language code
            
        Returns:
            Dictionary with transcript and metadata
        """
        segments = self.transcribe_audio(audio_path, language)
        
        return {
            'video_name': video_name,
            'language': language or 'auto-detected',
            'num_segments': len(segments),
            'total_duration': segments[-1].end_time if segments else 0,
            'segments': [seg.to_dict() for seg in segments]
        }


# Global transcriber instance (singleton pattern)
_transcriber_instance = None


def get_transcriber(model_name: str = "base") -> WhisperTranscriber:
    """
    Get or create global transcriber instance.
    Avoids loading model multiple times.
    
    Args:
        model_name: Whisper model size
        
    Returns:
        WhisperTranscriber instance
    """
    global _transcriber_instance
    
    if _transcriber_instance is None or _transcriber_instance.model_name != model_name:
        _transcriber_instance = WhisperTranscriber(model_name)
        _transcriber_instance.load_model()
    
    return _transcriber_instance


if __name__ == '__main__':
    # Test transcription
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        
        print(f"Transcribing: {audio_file}")
        
        transcriber = get_transcriber("base")
        segments = transcriber.transcribe_audio(audio_file)
        
        print(f"\nGenerated {len(segments)} segments:\n")
        
        for i, seg in enumerate(segments[:5], 1):
            print(f"Segment {i}:")
            print(f"  Time: {seg.start_time:.2f}s - {seg.end_time:.2f}s")
            print(f"  Text: {seg.text}")
            print()
        
        if len(segments) > 5:
            print(f"... and {len(segments) - 5} more segments")
    else:
        print("Usage: python transcription.py <audio_file>")
