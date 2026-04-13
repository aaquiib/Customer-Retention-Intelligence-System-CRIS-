"""FastAPI application entry point and model cache."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.pipeline import InferencePipeline
from inference.business_rules import RetentionActionDecider
from inference.shap_explainer import SHAPExplainer
from src.config import get_config
from src.utils import setup_logging

# Configure logging
setup_logging(get_config())
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# MODEL CACHE (SINGLETON)
# ─────────────────────────────────────────────────────────────────

class ModelCache:
    """Singleton cache for models - loads once at startup."""
    
    _instance: Optional['ModelCache'] = None
    
    def __init__(self):
        """Initialize model cache."""
        self.pipeline: Optional[InferencePipeline] = None
        self.action_decider: Optional[RetentionActionDecider] = None
        self.shap_explainer: Optional[SHAPExplainer] = None
        self.config = None
        self.is_loaded = False
    
    @classmethod
    def get_instance(cls) -> 'ModelCache':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = ModelCache()
        return cls._instance
    
    def load_models(self) -> None:
        """Load all models at startup."""
        if self.is_loaded:
            logger.info("Models already loaded")
            return
        
        try:
            logger.info("Loading models into cache...")
            self.config = get_config()
            self.pipeline = InferencePipeline(self.config)
            self.action_decider = RetentionActionDecider("config/business_rules.json")
            
            # Initialize SHAP explainer (non-blocking, warnings logged if fails)
            try:
                logger.info("Initializing SHAP explainer...")
                self.shap_explainer = SHAPExplainer(
                    pipeline=self.pipeline,
                    background_sample_path=None,
                    explainer_type="tree",
                    n_background_samples=200
                )
                logger.info("✓ SHAP explainer initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize SHAP explainer (optional): {e}")
                self.shap_explainer = None
            
            self.is_loaded = True
            logger.info("✓ All models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def unload_models(self) -> None:
        """Unload models on shutdown."""
        self.pipeline = None
        self.action_decider = None
        self.shap_explainer = None
        self.is_loaded = False
        logger.info("Models unloaded")


# ─────────────────────────────────────────────────────────────────
# LIFECYCLE EVENTS
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup/shutdown lifecycle."""
    # Startup
    logger.info("Starting CRIS API server...")
    model_cache = ModelCache.get_instance()
    model_cache.load_models()
    
    yield
    
    # Shutdown
    logger.info("Shutting down CRIS API server...")
    model_cache.unload_models()


# ─────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="CRIS - Customer Retention Intelligence System",
        description="Inference API for churn prediction and customer retention actions",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configure CORS for dashboard access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ─────────────────────────────────────────────────────────────────
    # HEALTH CHECK ENDPOINTS
    # ─────────────────────────────────────────────────────────────────
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        model_cache = ModelCache.get_instance()
        return {
            "status": "healthy" if model_cache.is_loaded else "initializing",
            "models_loaded": model_cache.is_loaded
        }
    
    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "CRIS API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health"
        }
    
    # ─────────────────────────────────────────────────────────────────
    # REGISTER ENDPOINTS
    # ─────────────────────────────────────────────────────────────────
    
    # Import and include endpoint routers
    from api.endpoints.predictions import router as predictions_router
    from api.endpoints.explanations import router as explanations_router
    from api.endpoints.whatif import router as whatif_router
    
    app.include_router(predictions_router, prefix="/api", tags=["predictions"])
    app.include_router(explanations_router, prefix="/api", tags=["explanations"])
    app.include_router(whatif_router, prefix="/api", tags=["what-if"])
    
    # ─────────────────────────────────────────────────────────────────
    # EXCEPTION HANDLERS
    # ─────────────────────────────────────────────────────────────────
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Invalid input", "detail": str(exc)}
        )
    
    @app.exception_handler(Exception)
    async def general_error_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error"}
        )
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        reload=False  # Set to True for development
    )
