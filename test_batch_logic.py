import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from app.services.embedding_service import EmbeddingService, EmbeddingOutput
from app.services.vector_store import VectorStoreService

class TestBatchLogic(unittest.TestCase):
    @patch('app.services.embedding_service.BGEM3FlagModel')
    def setUp(self, mock_model_class):
        # Mock the FlagModel to avoid loading it
        self.mock_model = MagicMock()
        mock_model_class.return_value = self.mock_model
        self.embedding_service = EmbeddingService()
        
    @patch('app.services.vector_store.QdrantClient')
    def test_batch_embedding_generation(self, mock_qdrant_client):
        # Mock model's encode method
        texts = ["text 1", "text 2"]
        mock_output = {
            'dense_vecs': [np.random.rand(1024), np.random.rand(1024)],
            'lexical_weights': [{"1": 0.5}, {"2": 0.6}],
            'colbert_vecs': [np.random.rand(5, 1024), np.random.rand(5, 1024)]
        }
        self.mock_model.encode.return_value = mock_output
        
        results = self.embedding_service.generate_batch_embeddings(texts)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], EmbeddingOutput)
        self.assertTrue(np.array_equal(results[0].dense_vector, mock_output['dense_vecs'][0]))
        self.assertEqual(results[0].sparse_weights, mock_output['lexical_weights'][0])
        
    @patch('app.services.vector_store.QdrantClient')
    def test_batch_upsert(self, mock_qdrant_client_class):
        mock_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_client
        vector_service = VectorStoreService()
        
        from qdrant_client import models
        points = [
            models.PointStruct(
                id="1",
                payload={"name": "test"},
                vector={"dense": [0.1]*1024}
            )
        ]
        
        vector_service.batch_upsert(points)
        
        # Verify that client.upsert was called with points and wait=True
        mock_client.upsert.assert_called_once()
        args, kwargs = mock_client.upsert.call_args
        self.assertEqual(kwargs['points'], points)
        self.assertTrue(kwargs['wait'])

if __name__ == '__main__':
    unittest.main()
