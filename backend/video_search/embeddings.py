"""
Text embedding generation using sentence-transformers.
Converts transcript text to dense vector embeddings for semantic search.

IMPORTANT: This uses LOCAL models only - no online API calls.
All models are downloaded once and cached locally.
No API keys or internet connection needed after initial download.
"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Generates embeddings for text using sentence-transformers.
    Uses local models for offline operation.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize text embedder.
        
        Args:
            model_name: Sentence transformer model name
                - 'all-MiniLM-L6-v2': Fast, good quality, 384 dims (RECOMMENDED)
                - 'all-mpnet-base-v2': Better quality, 768 dims, slower
                - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        logger.info(f"Initializing text embedder with model: {model_name}")
    
    def load_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            List of floats representing the embedding vector
        """
        if self.model is None:
            self.load_model()
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_tensor=False)
        
        # Convert to list for JSON serialization
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batched for efficiency).
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Batch encode for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 10,
            convert_to_tensor=False
        )
        
        # Convert to list of lists
        return [emb.tolist() for emb in embeddings]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self.embedding_dim is None:
            self.load_model()
        return self.embedding_dim


# Global embedder instance (singleton pattern)
_embedder_instance = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> TextEmbedder:
    """
    Get or create global embedder instance.
    Avoids loading model multiple times.
    
    Args:
        model_name: Sentence transformer model name
        
    Returns:
        TextEmbedder instance
    """
    global _embedder_instance
    
    if _embedder_instance is None or _embedder_instance.model_name != model_name:
        _embedder_instance = TextEmbedder(model_name)
        _embedder_instance.load_model()
    
    return _embedder_instance


if __name__ == '__main__':
    # Test embedding generation
    embedder = get_embedder()
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn canine leaps above an idle hound",
        "Machine learning is revolutionizing artificial intelligence",
        "What is the weather like today?"
    ]
    
    print(f"Embedding model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    print()
    
    # Single embedding
    print("Single text embedding:")
    embedding = embedder.embed_text(test_texts[0])
    print(f"  Text: {test_texts[0]}")
    print(f"  Embedding (first 10 dims): {embedding[:10]}")
    print()
    
    # Batch embeddings
    print("Batch embeddings:")
    embeddings = embedder.embed_texts(test_texts)
    print(f"  Generated {len(embeddings)} embeddings")
    print(f"  Shape: {len(embeddings)} x {len(embeddings[0])}")
