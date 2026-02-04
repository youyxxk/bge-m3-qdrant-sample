"""
FastAPI application entry point for RAG Service.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import settings
from app.api.v1.endpoints import router as v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs on startup and shutdown.
    """
    # Startup
    print(f"Starting RAG Service...")
    print(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"Collection: {settings.collection_name}")
    print(f"Model: {settings.model_name}")
    
    yield
    
    # Shutdown
    print("Shutting down RAG Service...")


app = FastAPI(
    title="RAG Service API",
    description="Hybrid Search RAG Backend using BGE-M3 and Qdrant",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Include API v1 router
app.include_router(
    v1_router,
    prefix=settings.api_v1_prefix,
    tags=["RAG"]
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint for health check."""
    return {
        "service": "RAG Service",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
