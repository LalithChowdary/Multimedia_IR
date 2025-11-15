"""
Main search interface for video search engine.
Orchestrates the full pipeline: extract → transcribe → embed → index/search.
"""

import os
import uuid
import logging
import json
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

try:
    # Try relative imports (when used as module)
    from .audio_extractor import extract_audio_from_video, get_video_info, cleanup_temp_audio
    from .transcription import get_transcriber, TranscriptSegment
    from .embeddings import get_embedder
    from .vector_db import get_vector_db
except ImportError:
    # Fallback to absolute imports (when run as script)
    from audio_extractor import extract_audio_from_video, get_video_info, cleanup_temp_audio
    from transcription import get_transcriber, TranscriptSegment
    from embeddings import get_embedder
    from vector_db import get_vector_db

logger = logging.getLogger(__name__)

# Configuration
WHISPER_MODEL = "base"  # Can be: tiny, base, small, medium, large
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # LOCAL embedding model (no online API needed)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, 'database', 'transcripts')
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)


class VideoSearchEngine:
    """
    Complete video search engine.
    Handles indexing and searching of video transcripts.
    """
    
    def __init__(
        self,
        whisper_model: str = WHISPER_MODEL,
        embedding_model: str = EMBEDDING_MODEL
    ):
        """
        Initialize search engine components.
        
        Args:
            whisper_model: Whisper model size for transcription
            embedding_model: Sentence transformer model for embeddings
        """
        self.transcriber = get_transcriber(whisper_model)
        self.embedder = get_embedder(embedding_model)
        self.vector_db = get_vector_db()
        
        logger.info("Video search engine initialized")
    
    def index_video(
        self,
        video_path: str,
        video_id: Optional[str] = None,
        language: Optional[str] = None,
        chunk_length: int = 30
    ) -> Dict:
        """
        Index a video: extract audio → transcribe → embed → store.
        
        Args:
            video_path: Path to the video file
            video_id: Optional unique ID (auto-generated if not provided)
            language: Optional language code for transcription
            chunk_length: Target length for transcript chunks in seconds
            
        Returns:
            Dictionary with indexing results and metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Generate video ID if not provided
        if video_id is None:
            video_id = str(uuid.uuid4())
        
        video_name = os.path.basename(video_path)
        logger.info(f"Starting indexing for video: {video_name}")
        
        try:
            # Step 1: Extract audio from video
            logger.info("Step 1: Extracting audio...")
            audio_path = extract_audio_from_video(video_path)
            
            # Step 2: Transcribe audio with Whisper
            logger.info("Step 2: Transcribing audio...")
            segments = self.transcriber.transcribe_audio(
                audio_path,
                language=language,
                chunk_length=chunk_length
            )
            
            if not segments:
                raise ValueError("No transcript segments generated")
            
            # Step 3: Generate embeddings for transcript segments
            logger.info("Step 3: Generating embeddings...")
            segment_texts = [seg.text for seg in segments]
            embeddings = self.embedder.embed_texts(segment_texts)
            
            # Step 4: Save transcript to text file
            logger.info("Step 4: Saving transcript to file...")
            transcript_path = self._save_transcript(
                video_id=video_id,
                video_name=video_name,
                segments=segments
            )
            
            # Step 5: Store in vector database
            logger.info("Step 5: Storing in vector database...")
            segment_dicts = [seg.to_dict() for seg in segments]
            self.vector_db.add_video_transcripts(
                video_id=video_id,
                video_name=video_name,
                segments=segment_dicts,
                embeddings=embeddings
            )
            
            # Cleanup temporary audio file
            cleanup_temp_audio(audio_path)
            
            # Prepare result
            result = {
                'video_id': video_id,
                'video_name': video_name,
                'video_path': video_path,
                'num_segments': len(segments),
                'total_duration': segments[-1].end_time if segments else 0,
                'language': language or 'auto-detected',
                'transcript_file': transcript_path,
                'status': 'success'
            }
            
            logger.info(f"Video indexed successfully: {video_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to index video {video_name}: {e}")
            raise
    
    def _save_transcript(
        self,
        video_id: str,
        video_name: str,
        segments: List[TranscriptSegment]
    ) -> str:
        """
        Save transcript to text file in database/transcripts.
        
        Args:
            video_id: Unique video identifier
            video_name: Name of the video file
            segments: List of transcript segments
            
        Returns:
            Path to saved transcript file
        """
        # Create filename from video_id
        transcript_filename = f"{video_id}.txt"
        transcript_path = os.path.join(TRANSCRIPTS_DIR, transcript_filename)
        
        # Also save JSON version with full metadata
        json_filename = f"{video_id}.json"
        json_path = os.path.join(TRANSCRIPTS_DIR, json_filename)
        
        try:
            # Save plain text transcript
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcript for: {video_name}\n")
                f.write(f"Video ID: {video_id}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total segments: {len(segments)}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, seg in enumerate(segments, 1):
                    timestamp = f"[{seg.start_time:.1f}s - {seg.end_time:.1f}s]"
                    f.write(f"Segment {i} {timestamp}:\n")
                    f.write(f"{seg.text}\n\n")
            
            # Save JSON version with full metadata
            json_data = {
                'video_id': video_id,
                'video_name': video_name,
                'generated_at': datetime.now().isoformat(),
                'num_segments': len(segments),
                'total_duration': segments[-1].end_time if segments else 0,
                'segments': [seg.to_dict() for seg in segments]
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transcript saved to: {transcript_path}")
            logger.info(f"JSON metadata saved to: {json_path}")
            
            return transcript_path
            
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")
            raise
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        video_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for videos matching the query.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            video_filter: Optional video_id to search within specific video
            
        Returns:
            List of search results with video name, timestamp, and text
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Searching for: '{query}'")
        
        try:
            # Step 1: Generate embedding for query
            query_embedding = self.embedder.embed_text(query)
            
            # Step 2: Search vector database
            results = self.vector_db.search(
                query_embedding=query_embedding,
                n_results=n_results,
                video_filter=video_filter
            )
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def list_indexed_videos(self) -> List[Dict]:
        """
        List all indexed videos.
        
        Returns:
            List of videos with metadata
        """
        return self.vector_db.list_videos()
    
    def delete_video(self, video_id: str):
        """
        Remove a video from the index.
        
        Args:
            video_id: ID of the video to remove
        """
        logger.info(f"Deleting video: {video_id}")
        self.vector_db.delete_video(video_id)
    
    def get_stats(self) -> Dict:
        """
        Get search engine statistics.
        
        Returns:
            Dictionary with stats
        """
        return self.vector_db.get_stats()


# Global search engine instance
_search_engine = None


def get_search_engine(
    whisper_model: str = WHISPER_MODEL,
    embedding_model: str = EMBEDDING_MODEL
) -> VideoSearchEngine:
    """
    Get or create global search engine instance.
    
    Args:
        whisper_model: Whisper model size
        embedding_model: Sentence transformer model
        
    Returns:
        VideoSearchEngine instance
    """
    global _search_engine
    
    if _search_engine is None:
        _search_engine = VideoSearchEngine(whisper_model, embedding_model)
    
    return _search_engine


if __name__ == '__main__':
    # Test search engine
    import sys
    
    engine = get_search_engine()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'index' and len(sys.argv) > 2:
            video_path = sys.argv[2]
            print(f"Indexing video: {video_path}")
            result = engine.index_video(video_path)
            print(f"\nIndexing complete!")
            print(f"  Video ID: {result['video_id']}")
            print(f"  Segments: {result['num_segments']}")
            print(f"  Duration: {result['total_duration']:.1f}s")
        
        elif command == 'search' and len(sys.argv) > 2:
            query = ' '.join(sys.argv[2:])
            print(f"Searching for: '{query}'")
            results = engine.search(query, n_results=5)
            
            print(f"\nFound {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['video_name']}")
                print(f"   Time: {result['start_time']:.1f}s - {result['end_time']:.1f}s")
                print(f"   Score: {result['score']:.3f}")
                print(f"   Text: {result['text']}")
                print()
        
        elif command == 'list':
            videos = engine.list_indexed_videos()
            print(f"Indexed videos ({len(videos)}):\n")
            for video in videos:
                print(f"  - {video['video_name']}: {video['segment_count']} segments")
        
        elif command == 'stats':
            stats = engine.get_stats()
            print("Search Engine Statistics:")
            print(f"  Total videos: {stats['total_videos']}")
            print(f"  Total segments: {stats['total_segments']}")
        
        else:
            print("Usage:")
            print("  python search.py index <video_file>")
            print("  python search.py search <query>")
            print("  python search.py list")
            print("  python search.py stats")
    else:
        print("Video Search Engine")
        print("\nUsage:")
        print("  python search.py index <video_file>   - Index a video")
        print("  python search.py search <query>       - Search videos")
        print("  python search.py list                 - List indexed videos")
        print("  python search.py stats                - Show statistics")
