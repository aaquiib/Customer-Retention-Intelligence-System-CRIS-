"""FastAPI application entry point and model cache."""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.pipeline import InferencePipeline
from inference.business_rules import RetentionActionDecider
from inference.shap_explainer import SHAPExplainer
from src.config import get_config
from src.utils import setup_logging

# Load environment variables from root directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

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
        self._global_shap_cache: Optional[List[Dict[str, Any]]] = None
    
    @classmethod
    def get_instance(cls) -> 'ModelCache':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = ModelCache()
        return cls._instance
    
    def _compute_global_shap(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Pre-compute global SHAP feature importance on background samples.
        
        This is called once at startup to cache the computation.
        Later requests use the cached result instead of recomputing.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            List of dicts: [{"feature_name": str, "importance": float, "sign": str}, ...]
            Empty list if SHAP explainer not available or computation fails.
        """
        if not self.shap_explainer or not self.shap_explainer.explainer:
            logger.warning("SHAP explainer not available, skipping global SHAP cache")
            return []
        
        try:
            t0 = time.time()
            logger.info("Pre-computing global SHAP on background samples...")
            
            # Get background samples and explainer
            background_X = self.shap_explainer.background_X
            explainer = self.shap_explainer.explainer
            
            if background_X is None:
                logger.warning("Background data not available for SHAP computation")
                return []
            
            # Compute SHAP values for background samples
            shap_vals = explainer.shap_values(background_X)
            
            # Handle binary classifier output: use positive class (class 1)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            logger.info(f"SHAP background computation took {time.time()-t0:.2f}s")
            
            # Compute mean absolute SHAP per feature
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            
            # Get feature names
            feature_names = self.shap_explainer.feature_names
            if not feature_names:
                logger.warning("Feature names not available from SHAP explainer")
                return []
            
            # Build top-N features list
            top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
            
            top_features = []
            for idx in top_indices:
                feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                importance = float(mean_abs_shap[idx])
                
                # Determine sign: compute mean SHAP for this feature
                mean_shap = shap_vals[:, idx].mean()
                sign = "positive" if mean_shap > 0 else "negative"
                
                top_features.append({
                    "feature_name": str(feature_name),
                    "importance": importance,
                    "sign": sign
                })
            
            logger.info(f"Global SHAP cache computed with top {len(top_features)} features")
            return top_features
        
        except Exception as e:
            logger.warning(f"Failed to compute global SHAP cache: {e}")
            return []
    
    def get_global_shap(self) -> List[Dict[str, Any]]:
        """
        Get pre-computed global SHAP feature importance.
        
        Returns the cached result computed at startup.
        
        Returns:
            List of dicts with feature importance, or empty list if not available.
        """
        return self._global_shap_cache if self._global_shap_cache is not None else []
    
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
            
            # Pre-compute global SHAP feature importance on background samples
            logger.info("Pre-computing global SHAP on background samples...")
            self._global_shap_cache = self._compute_global_shap(top_n=10)
            logger.info("Global SHAP cache ready.")
        
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
    logger.info("ModelCache warm — global SHAP pre-computed. Ready to serve requests.")
    
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
    
    # Configure CORS for frontend access (supports both local and production URLs)
    # For Render: set FRONTEND_URL via environment variables
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
    # Support multiple URLs separated by comma (e.g., "http://localhost:8501,https://myapp.onrender.com")
    frontend_urls = [url.strip() for url in frontend_url.split(",") if url.strip()]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=frontend_urls,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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
    
    # Render uses PORT env var, fallback to API_PORT for local dev
    port = int(os.getenv("PORT") or os.getenv("API_PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        reload=False  # Set to True for development
    )
