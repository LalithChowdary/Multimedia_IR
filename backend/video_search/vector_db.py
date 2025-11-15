"""
Vector database for storing and searching video transcript embeddings.
Uses ChromaDB for efficient similarity search.
"""

import os
import logging
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Database configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'database', 'video_search')


class VideoVectorDB:
    """
    Vector database for video transcript search using ChromaDB.
    Stores embeddings with metadata for retrieval.
    """
    
    def __init__(self, collection_name: str = "video_transcripts"):
        """
        Initialize vector database.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        logger.info(f"Initializing vector database: {collection_name}")
    
    def initialize(self):
        """Initialize ChromaDB client and collection."""
        if self.client is None:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Video transcript embeddings for semantic search"}
            )
            
            logger.info(f"Vector database initialized at: {DB_PATH}")
            logger.info(f"Collection '{self.collection_name}' ready")
    
    def add_video_transcripts(
        self,
        video_id: str,
        video_name: str,
        segments: List[Dict],
        embeddings: List[List[float]]
    ):
        """
        Add video transcript segments to the database.
        
        Args:
            video_id: Unique identifier for the video
            video_name: Name of the video file
            segments: List of transcript segments with timestamps
            embeddings: List of embedding vectors for each segment
        """
        if self.collection is None:
            self.initialize()
        
        if len(segments) != len(embeddings):
            raise ValueError("Number of segments must match number of embeddings")
        
        logger.info(f"Adding {len(segments)} segments for video: {video_name}")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for i, (segment, embedding) in enumerate(zip(segments, embeddings)):
            # Create unique ID for each segment
            segment_id = f"{video_id}_seg_{i}"
            ids.append(segment_id)
            
            # Document is the transcript text
            documents.append(segment['text'])
            
            # Metadata includes video info and timestamps
            metadata = {
                'video_id': video_id,
                'video_name': video_name,
                'segment_index': i,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'duration': segment['end_time'] - segment['start_time']
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {len(segments)} segments to database")
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        video_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar transcript segments using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            n_results: Number of results to return
            video_filter: Optional video_id to filter results
            
        Returns:
            List of matching segments with metadata and scores
        """
        if self.collection is None:
            self.initialize()
        
        # Build filter if specified
        where_filter = None
        if video_filter:
            where_filter = {"video_id": video_filter}
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        
        if results and results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                result = {
                    'video_name': results['metadatas'][0][i]['video_name'],
                    'video_id': results['metadatas'][0][i]['video_id'],
                    'text': results['documents'][0][i],
                    'start_time': results['metadatas'][0][i]['start_time'],
                    'end_time': results['metadatas'][0][i]['end_time'],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'segment_index': results['metadatas'][0][i]['segment_index']
                }
                formatted_results.append(result)
        
        logger.info(f"Search returned {len(formatted_results)} results")
        return formatted_results
    
    def delete_video(self, video_id: str):
        """
        Delete all segments for a specific video.
        
        Args:
            video_id: ID of the video to delete
        """
        if self.collection is None:
            self.initialize()
        
        # Get all IDs for this video
        results = self.collection.get(
            where={"video_id": video_id}
        )
        
        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} segments for video: {video_id}")
        else:
            logger.warning(f"No segments found for video: {video_id}")
    
    def list_videos(self) -> List[Dict]:
        """
        List all videos in the database.
        
        Returns:
            List of unique videos with metadata
        """
        if self.collection is None:
            self.initialize()
        
        # Get all documents
        results = self.collection.get()
        
        if not results or not results['metadatas']:
            return []
        
        # Extract unique videos
        videos_dict = {}
        for metadata in results['metadatas']:
            video_id = metadata['video_id']
            if video_id not in videos_dict:
                videos_dict[video_id] = {
                    'video_id': video_id,
                    'video_name': metadata['video_name'],
                    'segment_count': 0
                }
            videos_dict[video_id]['segment_count'] += 1
        
        return list(videos_dict.values())
    
    def get_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database stats
        """
        if self.collection is None:
            self.initialize()
        
        total_segments = self.collection.count()
        videos = self.list_videos()
        
        return {
            'total_segments': total_segments,
            'total_videos': len(videos),
            'collection_name': self.collection_name,
            'videos': videos
        }
    
    def reset(self):
        """Reset the entire database (delete all data)."""
        if self.collection is None:
            self.initialize()
        
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Video transcript embeddings for semantic search"}
        )
        logger.warning("Database reset - all data deleted")


# Global database instance
_db_instance = None


def get_vector_db(collection_name: str = "video_transcripts") -> VideoVectorDB:
    """
    Get or create global vector database instance.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        VideoVectorDB instance
    """
    global _db_instance
    
    if _db_instance is None or _db_instance.collection_name != collection_name:
        _db_instance = VideoVectorDB(collection_name)
        _db_instance.initialize()
    
    return _db_instance


if __name__ == '__main__':
    # Test vector database
    db = get_vector_db()
    
    print("Vector Database Test")
    print("=" * 60)
    
    # Get stats
    stats = db.get_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Collection: {stats['collection_name']}")
    
    if stats['videos']:
        print(f"\nVideos in database:")
        for video in stats['videos']:
            print(f"  - {video['video_name']}: {video['segment_count']} segments")
