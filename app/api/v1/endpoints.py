"""
API endpoints for RAG service.
Uses dependency injection for EmbeddingService and VectorStoreService.
"""

import csv
import io
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
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
from app.utils.chunking import get_chunks


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
    vector_svc: VectorStoreDep,
    chunking_strategy: str = Query("none", description="Chunking strategy: none, character, word, recursive, semantic"),
    chunk_size: int = Query(500, description="Maximum size of chunks (chars/words)"),
    chunk_overlap: int = Query(50, description="Overlap between chunks"),
    threshold: float = Query(0.5, description="Similarity threshold for semantic chunking")
) -> IngestResponse:
    """
    Ingest a document into the vector store.
    
    - Generates dense, sparse, and ColBERT embeddings
    - Stores the document with all vector types in Qdrant
    """
    try:
        # Ensure collection exists
        vector_svc.create_collection_if_not_exists()
        
        # Chunking logic
        chunks = get_chunks(
            document.description, 
            strategy=chunking_strategy, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            google_api_key=settings.google_api_key,
            threshold=threshold,
            model_name=settings.google_embedding_model
        )
        
        points = []
        for idx, chunk_text in enumerate(chunks):
            # Format text for embedding
            text = embedding_svc.format_product_text(
                name=document.name,
                description=chunk_text
            )
            
            # Generate embeddings
            embeddings: EmbeddingOutput = embedding_svc.generate_embeddings(text)
            
            # Prepare payload
            payload: dict[str, Any] = document.model_dump()
            payload["description"] = chunk_text
            
            if len(chunks) > 1 or chunking_strategy != "none":
                try:
                    namespace_uuid = uuid.UUID(document.id)
                except ValueError:
                    namespace_uuid = uuid.NAMESPACE_OID
                
                chunk_id = str(uuid.uuid5(namespace_uuid, f"{document.id}_chunk_{idx}"))
                payload["id"] = chunk_id
                payload["parent_id"] = document.id
                payload["chunk_index"] = idx
                payload["total_chunks"] = len(chunks)
                payload["chunk_metadata"] = {
                    "strategy": chunking_strategy,
                    "size": chunk_size,
                    "overlap": chunk_overlap
                }
            else:
                chunk_id = document.id
                
            qdrant_sparse = vector_svc.create_sparse_vector(embeddings.sparse_weights)
            vector_data = {
                "dense": embeddings.dense_vector.tolist(),
                "sparse": qdrant_sparse
            }
            if embeddings.colbert_vectors is not None:
                vector_data["colbert"] = embeddings.colbert_vectors.tolist()

            points.append(
                models.PointStruct(
                    id=chunk_id,
                    payload=payload,
                    vector=vector_data
                )
            )
            
        vector_svc.batch_upsert(points)
        
        return IngestResponse(
            success=True,
            message=f"Document ingested successfully into {len(chunks)} chunk(s)",
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
    file: UploadFile = File(...),
    chunking_strategy: str = Query("none", description="Chunking strategy: none, character, word, recursive, semantic"),
    chunk_size: int = Query(500, description="Maximum size of chunks (chars/words)"),
    chunk_overlap: int = Query(50, description="Overlap between chunks"),
    threshold: float = Query(0.5, description="Similarity threshold for semantic chunking")
) -> dict:
    """
    Ingest multiple documents from a pipe-separated CSV file.
    
    - Parses CSV (pipe-separated)
    - Validates each row with DocumentIngest schema
    - Applies text chunking on descriptions
    - Generates embeddings in batches
    - Upserts to Qdrant in batches
    """
    try:
        content = await file.read()
        stream = io.StringIO(content.decode("utf-8"))
        reader = csv.DictReader(stream, delimiter="|")
        
        # Ensure collection exists
        vector_svc.create_collection_if_not_exists()
        
        documents_to_embed = []
        original_doc_count = 0
        
        for row in reader:
            try:
                # Map CSV fields to DocumentIngest (handle case sensitivity if needed)
                doc_data = {
                    "id": row.get("Id"),
                    "name": row.get("Name"),
                    "description": row.get("Description", ""),
                    "price": float(row.get("Price", 0)),
                    "price_currency": row.get("PriceCurrency"),
                    "supply_ability": int(row.get("SupplyAbility")) if row.get("SupplyAbility") else None,
                    "minimum_order": int(row.get("MinimumOrder")) if row.get("MinimumOrder") else None
                }
                validated_doc = DocumentIngest(**doc_data)
                original_doc_count += 1
                
                chunks = get_chunks(
                    validated_doc.description, 
                    strategy=chunking_strategy, 
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap,
                    google_api_key=settings.google_api_key,
                    threshold=threshold,
                    model_name=settings.google_embedding_model
                )
                
                for idx, chunk_text in enumerate(chunks):
                    chunk_payload_data = validated_doc.model_dump()
                    chunk_payload_data["description"] = chunk_text
                    
                    if len(chunks) > 1 or chunking_strategy != "none":
                        try:
                            namespace_uuid = uuid.UUID(validated_doc.id)
                        except ValueError:
                            namespace_uuid = uuid.NAMESPACE_OID
                            
                        chunk_id = str(uuid.uuid5(namespace_uuid, f"{validated_doc.id}_chunk_{idx}"))
                        chunk_payload_data["id"] = chunk_id
                        chunk_payload_data["parent_id"] = validated_doc.id
                        chunk_payload_data["chunk_index"] = idx
                        chunk_payload_data["total_chunks"] = len(chunks)
                        chunk_payload_data["chunk_metadata"] = {
                            "strategy": chunking_strategy,
                            "size": chunk_size,
                            "overlap": chunk_overlap
                        }
                    else:
                        chunk_id = validated_doc.id
                        
                    documents_to_embed.append({
                        "id": chunk_id,
                        "name": validated_doc.name,
                        "description": chunk_text,
                        "payload": chunk_payload_data
                    })
            except (ValueError, TypeError) as e:
                print(f"Skipping invalid row: {row}. Error: {e}")
                continue

        if not documents_to_embed:
            return {"message": "No valid documents found in CSV", "count": 0}

        # Process in batches
        batch_size = 50
        total_chunks_ingested = len(documents_to_embed)
        
        for i in range(0, len(documents_to_embed), batch_size):
            batch = documents_to_embed[i:i + batch_size]
            
            # Prepare texts for embedding
            batch_texts = [
                embedding_svc.format_product_text(item["name"], item["description"])
                for item in batch
            ]
            
            # Generate embeddings in batch
            batch_embeddings = embedding_svc.generate_batch_embeddings(batch_texts)
            
            # Prepare PointStructs
            points = []
            for item, embs in zip(batch, batch_embeddings):
                qdrant_sparse = vector_svc.create_sparse_vector(embs.sparse_weights)
                vector_data = {
                    "dense": embs.dense_vector.tolist(),
                    "sparse": qdrant_sparse
                }
                if embs.colbert_vectors is not None:
                    vector_data["colbert"] = embs.colbert_vectors.tolist()

                points.append(
                    models.PointStruct(
                        id=item["id"],
                        payload=item["payload"],
                        vector=vector_data
                    )
                )
            
            # Batch upsert
            vector_svc.batch_upsert(points)

        return {
            "success": True,
            "message": f"Successfully ingested {original_doc_count} source documents into {total_chunks_ingested} chunks",
            "source_documents": original_doc_count,
            "total_chunks": total_chunks_ingested,
            "count": total_chunks_ingested
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

@router.get(
    "/list-google-models",
    summary="List Google AI embedding models",
    description="Fetch available embedding models from Google Generative AI"
)
async def list_models() -> dict:
    """List available Google AI models that support embedding."""
    from app.utils.chunking import list_google_models
    
    if not settings.google_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GOOGLE_API_KEY is not configured"
        )
    
    models = list_google_models(settings.google_api_key)
    return {"models": models}
