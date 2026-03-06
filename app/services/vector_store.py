"""
Vector store service for Qdrant operations.
Implements hybrid search using prefetch mechanism for Dense + Sparse vectors.
"""

from typing import Any, Optional
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse


class VectorStoreService:
    """Service for Qdrant vector database operations."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "products",
        dense_vector_size: int = 1024
    ):
        """
        Initialize the vector store service.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection
            dense_vector_size: Size of dense vectors (default 1024 for BGE-M3)
        """
        self._client = QdrantClient(host, port=port)
        self._collection_name = collection_name
        self._dense_vector_size = dense_vector_size
    
    def create_collection_if_not_exists(self) -> bool:
        """
        Create collection with appropriate vector configurations if it doesn't exist.
        
        Returns:
            True if collection was created, False if it already exists
        """
        try:
            # Check if collection exists
            collections = self._client.get_collections()
            existing_names = [c.name for c in collections.collections]
            
            if self._collection_name in existing_names:
                return False
            
            # Create collection with dense, sparse, and ColBERT configurations
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self._dense_vector_size,
                        distance=models.Distance.COSINE
                    ),
                    "colbert": models.VectorParams(
                        size=self._dense_vector_size,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True
                        )
                    )
                },
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {str(e)}")
    
    def create_sparse_vector(self, sparse_data: dict) -> models.SparseVector:
        """
        Convert BGE-M3 sparse output to Qdrant sparse vector format.
        
        Args:
            sparse_data: Dictionary of token IDs to weights from BGE-M3
            
        Returns:
            Qdrant SparseVector with indices and values
        """
        sparse_indices = []
        sparse_values = []
        
        for key, value in sparse_data.items():
            # Only process positive values
            if float(value) > 0:
                # Handle string keys (token IDs)
                if isinstance(key, str):
                    if key.isdigit():
                        key = int(key)
                    else:
                        continue
                
                sparse_indices.append(key)
                sparse_values.append(float(value))
        
        return models.SparseVector(
            indices=sparse_indices,
            values=sparse_values
        )
    
    def upsert(
        self,
        point_id: str,
        payload: dict[str, Any],
        dense_vector: np.ndarray,
        sparse_weights: dict,
        colbert_vectors: Optional[np.ndarray] = None
    ) -> None:
        """
        Insert or update a document with all three vector types.
        """
        # Convert sparse weights to Qdrant format
        qdrant_sparse = self.create_sparse_vector(sparse_weights)
        
        point = models.PointStruct(
            id=point_id,
            payload=payload,
            vector={
                "dense": dense_vector.tolist() if isinstance(dense_vector, np.ndarray) else dense_vector,
                "sparse": qdrant_sparse
            }
        )
        
        if colbert_vectors is not None:
            point.vector["colbert"] = colbert_vectors.tolist() if isinstance(colbert_vectors, np.ndarray) else colbert_vectors
        
        self.batch_upsert([point])

    def batch_upsert(self, points: list[models.PointStruct]) -> None:
        """
        Batch insert or update documents.
        
        Args:
            points: List of Qdrant PointStruct objects
        """
        if not points:
            return
            
        try:
            self._client.upsert(
                collection_name=self._collection_name,
                wait=True,
                points=points
            )
        except Exception as e:
            raise RuntimeError(f"Failed to batch upsert documents: {str(e)}")
    
    def search(
        self,
        dense_vector: np.ndarray,
        sparse_weights: dict,
        colbert_vectors: Optional[np.ndarray] = None,
        limit: int = 3,
        prefetch_limit: int = 6
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search using Qdrant's prefetch mechanism.
        
        Uses prefetch to combine Dense and Sparse vector searches,
        then reranks with ColBERT for final results.
        
        Args:
            dense_vector: Dense query vector
            sparse_weights: Sparse query weights
            colbert_vectors: ColBERT query vectors for reranking
            limit: Maximum number of final results
            prefetch_limit: Number of results to prefetch from each vector type
            
        Returns:
            List of search results with id, score, and payload
        """
        # Convert sparse weights to Qdrant format
        qdrant_sparse = self.create_sparse_vector(sparse_weights)
        
        # Set up prefetch for hybrid search (Dense + Sparse)
        prefetch = [
            models.Prefetch(
                query=qdrant_sparse,
                using="sparse",
                limit=prefetch_limit
            ),
            models.Prefetch(
                query=dense_vector.tolist() if isinstance(dense_vector, np.ndarray) else dense_vector,
                using="dense",
                limit=prefetch_limit
            )
        ]
        
        try:
            # Perform search (hybrid or purely dense+sparse depending on ColBERT availability)
            if colbert_vectors is not None:
                # Hybrid search with ColBERT reranking
                results = self._client.query_points(
                    self._collection_name,
                    prefetch=prefetch,
                    query=colbert_vectors.tolist() if isinstance(colbert_vectors, np.ndarray) else colbert_vectors,
                    using="colbert",
                    with_payload=True,
                    limit=limit,
                )
            else:
                # Hybrid search without ColBERT (Dense + Sparse)
                results = self._client.query_points(
                    self._collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    with_payload=True,
                    limit=limit,
                )
            
            # Convert to list of dicts
            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload
                }
                for point in results.points
            ]
        except Exception as e:
            raise RuntimeError(f"Search failed: {str(e)}")
    
    def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            True if deleted successfully
        """
        try:
            self._client.delete_collection(self._collection_name)
            return True
        except Exception:
            return False
    
    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self._collection_name
