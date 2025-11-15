"""
Demo: Complete video search workflow with local embeddings and transcript saving.

This script demonstrates:
1. Local Whisper transcription (no online API)
2. Local embedding generation (no online API)  
3. Transcript file saving in database/transcripts/
4. Vector database storage
5. Semantic search

Usage:
    python demo_video_search.py <path_to_video_file>
"""

import sys
import os

# Add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'video_search'))

def demo_video_search(video_path):
    """
    Demonstrate complete video search workflow.
    """
    print("="*80)
    print("Video Search Demo - LOCAL EMBEDDINGS & TRANSCRIPT SAVING")
    print("="*80)
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    from search import get_search_engine
    
    # Initialize search engine
    print("\n[1/5] Initializing search engine...")
    engine = get_search_engine()
    print("✓ Whisper model loaded (LOCAL - no API)")
    print("✓ Embedding model loaded (LOCAL - no API)")
    
    # Index the video
    print(f"\n[2/5] Indexing video: {os.path.basename(video_path)}")
    print("This will:")
    print("  - Extract audio from video")
    print("  - Transcribe with Whisper (LOCAL)")
    print("  - Generate embeddings (LOCAL)")
    print("  - Save transcript to database/transcripts/")
    print("  - Store in vector database")
    print("\nProcessing... (this may take a minute)")
    
    try:
        result = engine.index_video(video_path, language='en')
        
        print("\n✅ Video indexed successfully!")
        print(f"  Video ID: {result['video_id']}")
        print(f"  Segments: {result['num_segments']}")
        print(f"  Duration: {result['total_duration']:.1f}s")
        print(f"  Transcript saved to: {result['transcript_file']}")
        
        # Show transcript file location
        transcript_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'database',
            'transcripts'
        )
        print(f"\n📁 Transcript files saved in: {transcript_dir}")
        
        # List saved files
        txt_file = f"{result['video_id']}.txt"
        json_file = f"{result['video_id']}.json"
        print(f"  - {txt_file} (plain text)")
        print(f"  - {json_file} (JSON with metadata)")
        
    except Exception as e:
        print(f"\n❌ Failed to index video: {e}")
        return
    
    # Demonstrate search
    print("\n[3/5] Testing semantic search...")
    
    test_queries = [
        "introduction",
        "main topic",
        "conclusion"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        results = engine.search(query, n_results=2)
        
        if results:
            for i, res in enumerate(results[:2], 1):
                print(f"    Result {i}:")
                print(f"      Time: {res['start_time']:.1f}s - {res['end_time']:.1f}s")
                print(f"      Score: {res['score']:.3f}")
                print(f"      Text: {res['text'][:100]}...")
        else:
            print("    No results found")
    
    # Show stats
    print("\n[4/5] Database statistics...")
    stats = engine.get_stats()
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Total segments: {stats['total_segments']}")
    
    # Show transcript preview
    print("\n[5/5] Transcript preview...")
    transcript_path = result['transcript_file']
    
    if os.path.exists(transcript_path):
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print("\n" + "".join(lines[:15]))
            print("  ... (see full transcript in file)")
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    print("\nKey Points:")
    print("  ✓ All processing done LOCALLY (no online APIs)")
    print("  ✓ Transcripts saved in database/transcripts/")
    print("  ✓ Embeddings stored in local ChromaDB")
    print("  ✓ Ready for semantic search")
    print("\nTranscript location:")
    print(f"  {transcript_path}")
    print("="*80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python demo_video_search.py <path_to_video_file>")
        print("\nExample:")
        print("  python demo_video_search.py ../videos/lecture.mp4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    demo_video_search(video_file)
