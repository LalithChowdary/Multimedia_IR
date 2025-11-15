"""
Video Search Engine Module

This module implements a semantic video search system using:
- Whisper for speech-to-text transcription (local/offline)
- Sentence transformers for text embeddings
- ChromaDB for vector storage and similarity search

Features:
- Extract audio from videos
- Transcribe with timestamps
- Semantic search across transcripts
- Return video + timestamp results
"""

__version__ = "1.0.0"
