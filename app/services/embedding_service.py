"""
Embedding service for generating BGE-M3 embeddings.
Handles dense, sparse, and ColBERT vector generation.
"""

from dataclasses import dataclass

from typing import Optional
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from app.core.config import settings


@dataclass
class EmbeddingOutput:
    """Container for embedding outputs."""
    dense_vector: np.ndarray
    sparse_weights: dict[str, float]
    colbert_vectors: Optional[np.ndarray] = None


class EmbeddingService:
    """Service for generating BGE-M3 embeddings."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the BGE-M3 model to use
            use_fp16: Whether to use FP16 precision for faster inference
        """
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self._model_name = model_name
    
    def generate_embeddings(self, text: str) -> EmbeddingOutput:
        """
        Generate embeddings for a given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            EmbeddingOutput containing dense, sparse, and optionally ColBERT vectors
        """
        use_colbert = settings.use_colbert
        output = self._model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=use_colbert
        )
        
        return EmbeddingOutput(
            dense_vector=output['dense_vecs'][0],
            sparse_weights=output['lexical_weights'][0],
            colbert_vectors=output['colbert_vecs'][0] if use_colbert else None
        )
    
    def generate_batch_embeddings(self, texts: list[str]) -> list[EmbeddingOutput]:
        """
        Generate embeddings for a list of texts in batch.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of EmbeddingOutput objects
        """
        if not texts:
            return []
            
        use_colbert = settings.use_colbert
        outputs = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=use_colbert
        )
        
        result = []
        for i in range(len(texts)):
            result.append(
                EmbeddingOutput(
                    dense_vector=outputs['dense_vecs'][i],
                    sparse_weights=outputs['lexical_weights'][i],
                    colbert_vectors=outputs['colbert_vecs'][i] if use_colbert else None
                )
            )
        return result
    
    @staticmethod
    def format_product_text(name: str, description: str) -> str:
        """
        Format product information for embedding.
        
        Args:
            name: Product name
            description: Product description
            
        Returns:
            Formatted text string for embedding
        """
        return f"Product: {name}\nDescription: {description}"
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name
