"""
Quick test to verify video search installation.
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import whisper
        print("✓ Whisper")
    except ImportError as e:
        print(f"✗ Whisper: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✓ Sentence Transformers")
    except ImportError as e:
        print(f"✗ Sentence Transformers: {e}")
        return False
    
    try:
        import chromadb
        print("✓ ChromaDB")
    except ImportError as e:
        print(f"✗ ChromaDB: {e}")
        return False
    
    try:
        import moviepy.editor
        print("✓ MoviePy")
    except ImportError as e:
        print(f"✗ MoviePy: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    return True


def test_modules():
    """Test that our video_search modules load correctly."""
    print("\nTesting video_search modules...")
    
    # Add to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'video_search'))
    
    try:
        from transcription import get_transcriber
        print("✓ Transcription module")
    except Exception as e:
        print(f"✗ Transcription module: {e}")
        return False
    
    try:
        from embeddings import get_embedder
        print("✓ Embeddings module")
    except Exception as e:
        print(f"✗ Embeddings module: {e}")
        return False
    
    try:
        from vector_db import get_vector_db
        print("✓ Vector DB module")
    except Exception as e:
        print(f"✗ Vector DB module: {e}")
        return False
    
    try:
        from search import get_search_engine
        print("✓ Search module")
    except Exception as e:
        print(f"✗ Search module: {e}")
        return False
    
    return True


def test_initialization():
    """Test that components can be initialized."""
    print("\nTesting component initialization...")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'video_search'))
    
    try:
        from embeddings import get_embedder
        embedder = get_embedder()
        dim = embedder.get_embedding_dimension()
        print(f"✓ Text embedder initialized (dim: {dim})")
    except Exception as e:
        print(f"✗ Text embedder: {e}")
        return False
    
    try:
        from vector_db import get_vector_db
        db = get_vector_db()
        stats = db.get_stats()
        print(f"✓ Vector DB initialized (videos: {stats['total_videos']}, segments: {stats['total_segments']})")
    except Exception as e:
        print(f"✗ Vector DB: {e}")
        return False
    
    print("\nNote: Whisper model will be downloaded on first use (~140MB for 'base' model)")
    
    return True


if __name__ == '__main__':
    print("="*60)
    print("Video Search Engine - Installation Test")
    print("="*60)
    print()
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test modules
    if not test_modules():
        success = False
    
    # Test initialization
    if not test_initialization():
        success = False
    
    print()
    print("="*60)
    
    if success:
        print("✅ All tests passed! Video search engine is ready.")
        print()
        print("Next steps:")
        print("1. Start the server: uvicorn main:app --reload")
        print("2. Upload a video: POST /video/upload")
        print("3. Search: GET /video/search?q=your+query")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("="*60)
