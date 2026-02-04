"""
Pydantic V2 schemas for API request/response models.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


class DocumentIngest(BaseModel):
    """Schema for document ingestion request."""
    
    id: str = Field(..., description="Unique identifier for the document")
    name: str = Field(..., description="Product name")
    description: str = Field(..., description="Product description")
    price: float = Field(..., ge=0, description="Product price")
    price_currency: str = Field(..., description="Currency code (e.g., USD, EUR)")
    supply_ability: Optional[int] = Field(None, ge=0, description="Available supply")
    minimum_order: Optional[int] = Field(None, ge=0, description="Minimum order quantity")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "product-001",
                "name": "Running Shoes",
                "description": "Lightweight running shoes for athletes",
                "price": 99.99,
                "price_currency": "USD",
                "supply_ability": 500,
                "minimum_order": 1
            }
        }


class SearchRequest(BaseModel):
    """Schema for search request."""
    
    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(default=3, ge=1, le=100, description="Maximum number of results")
    prefetch_limit: int = Field(default=6, ge=1, le=200, description="Prefetch limit for hybrid search")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "running shoes for men",
                "limit": 3,
                "prefetch_limit": 6
            }
        }


class SearchResult(BaseModel):
    """Schema for a single search result."""
    
    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Relevance score")
    payload: dict[str, Any] = Field(..., description="Document payload/metadata")


class SearchResponse(BaseModel):
    """Schema for search response."""
    
    results: List[SearchResult] = Field(..., description="List of search results")
    count: int = Field(..., description="Number of results returned")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "id": "product-001",
                        "score": 0.95,
                        "payload": {
                            "name": "Running Shoes",
                            "description": "Lightweight running shoes",
                            "price": 99.99
                        }
                    }
                ],
                "count": 1
            }
        }


class IngestResponse(BaseModel):
    """Schema for ingestion response."""
    
    success: bool = Field(..., description="Whether ingestion was successful")
    message: str = Field(..., description="Response message")
    document_id: Optional[str] = Field(None, description="ID of the ingested document")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document ingested successfully",
                "document_id": "product-001"
            }
        }
