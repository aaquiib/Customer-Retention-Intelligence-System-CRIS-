"""API client for FastAPI backend with caching and error handling."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import requests

from config import API_BASE_URL, API_TIMEOUT, API_RETRIES

logger = logging.getLogger(__name__)


class APIClient:
    """HTTP client for Churn Segmentation Decision System API."""
    
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = API_TIMEOUT, retries: int = API_RETRIES):
        """Initialize API client.
        
        Args:
            base_url: Base URL for API endpoints
            timeout: Timeout for requests in seconds
            retries: Number of retry attempts
        """
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        
        # Simple Python-level cache instead of Streamlit caching
        self._cache = {}
        self._cache_ttl = {}
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Tuple[bool, Dict, str]:
        """Make HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (without base URL)
            **kwargs: Additional arguments for requests
            
        Returns:
            Tuple of (success, data, error_message)
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.retries):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs
                )
                
                # Log request
                logger.info(f"{method} {endpoint} -> {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        return True, response.json(), ""
                    except Exception as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        return False, {}, f"Invalid JSON response: {str(e)}"
                
                elif response.status_code == 503:
                    if attempt < self.retries - 1:
                        time.sleep(1)
                        continue
                    return False, {}, "Models not loaded - server initializing"
                
                else:
                    try:
                        error_detail = response.json().get("detail", response.text)
                    except:
                        error_detail = response.text
                    return False, {}, f"API Error ({response.status_code}): {error_detail}"
            
            except requests.exceptions.Timeout:
                if attempt < self.retries - 1:
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(1)
                    continue
                return False, {}, "API request timeout - server not responding"
            
            except requests.exceptions.ConnectionError:
                return False, {}, f"Cannot connect to API at {self.base_url}"
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return False, {}, f"Unexpected error: {str(e)}"
        
        return False, {}, "Max retries exceeded"
    
    # ─────────────────────────────────────────────────────────────
    # HEALTH & MODEL INFO
    # ─────────────────────────────────────────────────────────────
    
    def get_health(_self) -> Tuple[bool, Dict, str]:
        """Get API health status.
        
        Returns:
            Tuple of (success, data, error)
        """
        # Health endpoint is at root level, not under /api prefix
        url = "http://localhost:8000/health"
        try:
            response = _self.session.get(url, timeout=_self.timeout)
            if response.status_code == 200:
                try:
                    return True, response.json(), ""
                except Exception as e:
                    logger.error(f"Failed to parse health response: {e}")
                    return False, {}, f"Invalid JSON response: {str(e)}"
            else:
                return False, {}, f"Health check failed: HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            return False, {}, "Health check timeout"
        except Exception as e:
            return False, {}, f"Health check error: {str(e)}"
    
    def get_model_info(_self) -> Tuple[bool, Dict, str]:
        """Get model metadata and performance metrics.
        
        Returns:
            Tuple of (success, model_info_dict, error)
        """
        success, data, error = _self._request("GET", "/explanations/model-info")
        if success:
            return True, data.get("model_info", {}), ""
        return False, {}, error
    
    # ─────────────────────────────────────────────────────────────
    # PREDICTIONS
    # ─────────────────────────────────────────────────────────────
    
    def predict_single(
        self,
        customer: Dict[str, Any],
        return_features: bool = False
    ) -> Tuple[bool, Dict, str]:
        """Get churn prediction for single customer.
        
        Args:
            customer: Customer data dict (19 fields)
            return_features: Include engineered features in response
            
        Returns:
            Tuple of (success, prediction_dict, error)
        """
        payload = {
            "customer": customer,
            "return_features": return_features
        }
        success, data, error = self._request(
            "POST",
            "/predict",
            json=payload
        )
        if success:
            return True, data.get("prediction", {}), ""
        return False, {}, error
    
    def predict_batch(
        self,
        csv_bytes: bytes
    ) -> Tuple[bool, Dict, str]:
        """Score batch of customers via CSV upload.
        
        Args:
            csv_bytes: File bytes of CSV with 19 customer columns
            
        Returns:
            Tuple of (success, batch_result_dict, error)
        """
        files = {"file": ("batch.csv", csv_bytes, "text/csv")}
        success, data, error = self._request(
            "POST",
            "/predict-batch",
            files=files
        )
        if success:
            return True, data.get("predictions", []), ""
        return False, {}, error
    
    def get_batch_template(self) -> Tuple[bool, bytes, str]:
        """Download CSV template for batch predictions.
        
        Returns:
            Tuple of (success, csv_bytes, error)
        """
        try:
            response = requests.get(
                f"{self.base_url}/predict-batch/template",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return True, response.content, ""
            return False, b"", f"Failed to download template: {response.status_code}"
        except Exception as e:
            return False, b"", f"Template download error: {str(e)}"
    
    # ─────────────────────────────────────────────────────────────
    # EXPLANATIONS
    # ─────────────────────────────────────────────────────────────
    
    def get_global_importance(
        _self,
        top_n: int = 10
    ) -> Tuple[bool, Dict, str]:
        """Get global feature importance (SHAP).
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Tuple of (success, importance_dict, error)
        """
        success, data, error = _self._request(
            "GET",
            f"/feature-importance/global?top_n={top_n}"
        )
        if success:
            return True, data.get("explanation", {}), ""
        return False, {}, error
    
    def get_instance_importance(
        self,
        customer: Dict[str, Any],
        top_n: int = 5
    ) -> Tuple[bool, Dict, str]:
        """Get SHAP values for specific customer.
        
        Args:
            customer: Customer data dict
            top_n: Number of top contributing features
            
        Returns:
            Tuple of (success, shap_dict, error)
        """
        payload = customer
        success, data, error = self._request(
            "POST",
            f"/feature-importance/instance?top_n={top_n}",
            json=payload
        )
        if success:
            return True, data.get("explanation", {}), ""
        return False, {}, error
    
    def get_explanation_methods(self) -> Tuple[bool, Dict, str]:
        """List available explainability methods.
        
        Returns:
            Tuple of (success, methods_dict, error)
        """
        success, data, error = self._request("GET", "/explanations/methods")
        if success:
            return True, data.get("methods", {}), ""
        return False, {}, error
    
    # ─────────────────────────────────────────────────────────────
    # WHAT-IF SIMULATION
    # ─────────────────────────────────────────────────────────────
    
    def what_if_single(
        self,
        customer: Dict[str, Any],
        modifications: Dict[str, Any]
    ) -> Tuple[bool, Dict, str]:
        """Run what-if scenario for single customer.
        
        Args:
            customer: Original customer data
            modifications: Changes to apply
            
        Returns:
            Tuple of (success, scenario_result, error)
        """
        payload = {
            "customer": customer,
            "modifications": modifications
        }
        success, data, error = self._request(
            "POST",
            "/what-if",
            json=payload
        )
        if success:
            # API returns WhatIfResponse with original_prediction, modified_prediction, delta directly
            return True, data, ""
        return False, {}, error
    
    def what_if_batch(
        self,
        scenarios: List[Dict[str, Any]]
    ) -> Tuple[bool, Dict, str]:
        """Run multiple what-if scenarios.
        
        Args:
            scenarios: List of scenario dicts, max 100
            
        Returns:
            Tuple of (success, results, error)
        """
        payload = scenarios
        success, data, error = self._request(
            "POST",
            "/what-if/batch",
            json=payload
        )
        if success:
            return True, data.get("results", []), ""
        return False, {}, error
    
    def get_policy_scenarios(self) -> Tuple[bool, List[Dict], str]:
        """Get pre-defined retention policy scenarios.
        
        Returns:
            Tuple of (success, scenarios_list, error)
        """
        success, data, error = self._request("GET", "/what-if/policy-changes")
        if success:
            return True, data.get("scenarios", []), ""
        return False, {}, error
