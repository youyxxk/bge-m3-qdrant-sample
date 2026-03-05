"""
API endpoints for RAG service.
Uses dependency injection for EmbeddingService and VectorStoreService.
"""

import csv
import io
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from typing import Annotated, Any
from qdrant_client import models

from app.core.config import settings
from app.models.schemas import (
    DocumentIngest,
    SearchRequest,
    SearchResponse,
    SearchResult,
    IngestResponse,
)
from app.services.embedding_service import EmbeddingService, EmbeddingOutput
from app.services.vector_store import VectorStoreService


router = APIRouter()


# Dependency injection functions
def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    return EmbeddingService(
        model_name=settings.model_name,
        use_fp16=settings.use_fp16
    )


def get_vector_store() -> VectorStoreService:
    """Get vector store service instance."""
    return VectorStoreService(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.collection_name,
        dense_vector_size=settings.dense_vector_size
    )


# Type aliases for dependency injection
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
VectorStoreDep = Annotated[VectorStoreService, Depends(get_vector_store)]


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
    description="Ingest a document by generating embeddings and storing in Qdrant"
)
async def ingest_document(
    document: DocumentIngest,
    embedding_svc: EmbeddingServiceDep,
    vector_svc: VectorStoreDep
) -> IngestResponse:
    """
    Ingest a document into the vector store.
    
    - Generates dense, sparse, and ColBERT embeddings
    - Stores the document with all vector types in Qdrant
    """
    try:
        # Ensure collection exists
        vector_svc.create_collection_if_not_exists()
        
        # Format text for embedding
        text = embedding_svc.format_product_text(
            name=document.name,
            description=document.description
        )
        
        # Generate embeddings
        embeddings: EmbeddingOutput = embedding_svc.generate_embeddings(text)
        
        # Prepare payload
        payload: dict[str, Any] = document.model_dump()
        
        # Upsert to vector store
        vector_svc.upsert(
            point_id=document.id,
            payload=payload,
            dense_vector=embeddings.dense_vector,
            sparse_weights=embeddings.sparse_weights,
            colbert_vectors=embeddings.colbert_vectors
        )
        return IngestResponse(
            success=True,
            message="Document ingested successfully",
            document_id=document.id
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


@router.post(
    "/ingest/csv",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest documents from CSV",
    description="Upload a pipe-separated CSV file to ingest multiple documents"
)
async def ingest_csv(
    embedding_svc: EmbeddingServiceDep,
    vector_svc: VectorStoreDep,
    file: UploadFile = File(...)
) -> dict:
    """
    Ingest multiple documents from a pipe-separated CSV file.
    
    - Parses CSV (pipe-separated)
    - Validates each row with DocumentIngest schema
    - Generates embeddings in batches
    - Upserts to Qdrant in batches
    """
    try:
        content = await file.read()
        stream = io.StringIO(content.decode("utf-8"))
        reader = csv.DictReader(stream, delimiter="|")
        
        # Ensure collection exists
        vector_svc.create_collection_if_not_exists()
        
        documents = []
        for row in reader:
            try:
                # Map CSV fields to DocumentIngest (handle case sensitivity if needed)
                # The user's example shows: Id|Name|Description|Price|PriceCurrency|SupplyAbility|MinimumOrder
                doc_data = {
                    "id": row.get("Id"),
                    "name": row.get("Name"),
                    "description": row.get("Description"),
                    "price": float(row.get("Price", 0)),
                    "price_currency": row.get("PriceCurrency"),
                    "supply_ability": int(row.get("SupplyAbility")) if row.get("SupplyAbility") else None,
                    "minimum_order": int(row.get("MinimumOrder")) if row.get("MinimumOrder") else None
                }
                documents.append(DocumentIngest(**doc_data))
            except (ValueError, TypeError) as e:
                print(f"Skipping invalid row: {row}. Error: {e}")
                continue

        if not documents:
            return {"message": "No valid documents found in CSV", "count": 0}

        # Process in batches
        batch_size = 50
        total_ingested = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Prepare texts for embedding
            batch_texts = [
                embedding_svc.format_product_text(doc.name, doc.description)
                for doc in batch
            ]
            
            # Generate embeddings in batch
            batch_embeddings = embedding_svc.generate_batch_embeddings(batch_texts)
            
            # Prepare PointStructs
            points = []
            for doc, embs in zip(batch, batch_embeddings):
                qdrant_sparse = vector_svc.create_sparse_vector(embs.sparse_weights)
                points.append(
                    models.PointStruct(
                        id=doc.id,
                        payload=doc.model_dump(),
                        vector={
                            "dense": embs.dense_vector.tolist(),
                            "colbert": embs.colbert_vectors.tolist(),
                            "sparse": qdrant_sparse
                        }
                    )
                )
            
            # Batch upsert
            vector_svc.batch_upsert(points)
            total_ingested += len(batch)

        return {
            "success": True,
            "message": f"Successfully ingested {total_ingested} documents",
            "count": total_ingested
        }

    except Exception as e:
        print(f"CSV Ingestion Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest CSV: {str(e)}"
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search documents",
    description="Perform hybrid search using dense, sparse, and ColBERT vectors"
)
async def search(
    request: SearchRequest,
    embedding_svc: EmbeddingServiceDep,
    vector_svc: VectorStoreDep
) -> SearchResponse:
    """
    Search for documents using hybrid search.
    
    - Uses prefetch mechanism to combine dense and sparse vector searches
    - Reranks results using ColBERT for improved relevance
    """
    try:
        # Generate query embeddings
        embeddings = embedding_svc.generate_embeddings(request.query)
        
        # Perform hybrid search
        results = vector_svc.search(
            dense_vector=embeddings.dense_vector,
            sparse_weights=embeddings.sparse_weights,
            colbert_vectors=embeddings.colbert_vectors,
            limit=request.limit,
            prefetch_limit=request.prefetch_limit
        )
        
        # Convert to response model
        search_results = [
            SearchResult(
                id=result["id"],
                score=result["score"],
                payload=result["payload"]
            )
            for result in results
        ]
        
        return SearchResponse(
            results=search_results,
            count=len(search_results)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post(
    "/init-collection",
    summary="Initialize collection",
    description="Create the Qdrant collection if it doesn't exist"
)
async def init_collection(
    vector_svc: VectorStoreDep
) -> dict:
    """Initialize the Qdrant collection with proper vector configurations."""
    try:
        created = vector_svc.create_collection_if_not_exists()
        
        if created:
            return {"message": f"Collection '{vector_svc.collection_name}' created successfully"}
        else:
            return {"message": f"Collection '{vector_svc.collection_name}' already exists"}
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize collection: {str(e)}"
        )
