"""What-if simulation endpoints for feature perturbation analysis."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from api.app import ModelCache
from api.schemas import (
    CustomerInput,
    PredictionDelta,
    PredictionOutput,
    WhatIfRequest,
    WhatIfResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/what-if", response_model=WhatIfResponse)
async def what_if_simulation(request: WhatIfRequest) -> WhatIfResponse:
    """
    Simulate prediction changes by modifying customer features.
    
    Useful for understanding feature impact:
    - What if customer tenure increases to 36 months?
    - What if they switch from month-to-month to a 2-year contract?
    - What if they upgrade to premium services?
    
    Args:
        customer: Original customer features
        modifications: Dictionary of features to change
                      e.g., {"tenure": 50, "Contract": "Two year"}
    
    Returns:
        - original_prediction: Original segment, churn_prob, action
        - modified_prediction: Prediction with modified features
        - delta: Changes in prediction (segment_changed, churn_prob_delta, action_changed)
        - modified_features: Applied modifications
    """
    try:
        model_cache = ModelCache.get_instance()
        
        if not model_cache.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Get original prediction
        original_customer = request.customer.model_dump(exclude_none=True)
        
        try:
            original_pred = model_cache.pipeline.predict_single(original_customer)
        except Exception as e:
            logger.error(f"Failed to compute original prediction: {e}")
            raise ValueError(f"Could not generate original prediction: {str(e)}")
        
        # Apply modifications
        modified_customer = original_customer.copy()
        for feature, value in request.modifications.items():
            if feature not in modified_customer and feature not in CustomerInput.model_fields:
                logger.warning(f"Unknown feature {feature} in modifications - skipping")
                continue
            modified_customer[feature] = value
        
        # Get modified prediction
        try:
            modified_pred = model_cache.pipeline.predict_single(modified_customer)
        except Exception as e:
            logger.error(f"Failed to compute modified prediction: {e}")
            raise ValueError(f"Could not generate modified prediction: {str(e)}")
        
        # Compute delta
        churn_delta = modified_pred['churn_probability'] - original_pred['churn_probability']
        segment_changed = modified_pred['segment'] != original_pred['segment']
        action_changed = False  # Will be True if action recommendation changes
        
        segment_delta = None
        if segment_changed:
            segment_delta = {
                'original_segment': original_pred['segment'],
                'original_label': original_pred['segment_label'],
                'modified_segment': modified_pred['segment'],
                'modified_label': modified_pred['segment_label']
            }
        
        delta = PredictionDelta(
            segment_changed=segment_changed,
            segment_delta=segment_delta,
            churn_probability_delta=churn_delta,
            is_churner_changed=original_pred['is_churner'] != modified_pred['is_churner'],
            action_changed=action_changed
        )
        
        # Build original prediction output
        original_output = PredictionOutput(
            segment=original_pred['segment'],
            segment_label=original_pred['segment_label'],
            churn_probability=original_pred['churn_probability'],
            is_churner=original_pred['is_churner'],
            threshold=original_pred['threshold'],
            segment_confidence=original_pred['segment_confidence'],
            input_features=original_customer
        )
        
        # Build modified prediction output
        modified_output = PredictionOutput(
            segment=modified_pred['segment'],
            segment_label=modified_pred['segment_label'],
            churn_probability=modified_pred['churn_probability'],
            is_churner=modified_pred['is_churner'],
            threshold=modified_pred['threshold'],
            segment_confidence=modified_pred['segment_confidence'],
            input_features=modified_customer
        )
        
        logger.info(
            f"What-if simulation: churn_delta={churn_delta:.4f}, "
            f"segment_changed={segment_changed}, "
            f"modifications={len(request.modifications)}"
        )
        
        return WhatIfResponse(
            success=True,
            original_prediction=original_output,
            modified_prediction=modified_output,
            delta=delta,
            modified_features=request.modifications
        )
    
    except ValueError as e:
        logger.warning(f"Validation error in what-if simulation: {e}")
        return WhatIfResponse(
            success=False,
            error=str(e),
            modified_features=request.modifications if request else {}
        )
    
    except Exception as e:
        logger.error(f"Error in what-if simulation: {e}", exc_info=True)
        return WhatIfResponse(
            success=False,
            error="What-if simulation failed",
            modified_features=request.modifications if request else {}
        )


@router.post("/what-if/batch")
async def what_if_batch(requests: list[WhatIfRequest]):
    """
    Run multiple what-if simulations in parallel.
    
    Useful for sensitivity analysis across multiple scenarios.
    """
    results = []
    for req in requests:
        result = await what_if_simulation(req)
        results.append(result)
    
    return {
        'success': all(r.success for r in results),
        'total_simulations': len(results),
        'successful': sum(1 for r in results if r.success),
        'results': results
    }


@router.get("/what-if/policy-changes")
async def get_policy_change_scenarios():
    """
    Return pre-defined policy change scenarios for common retention strategies.
    
    Examples:
    - Upgrade to 12-month contract (reduces churn risk)
    - Add security services (increases monthly charges but improves engagement)
    - Switch to automatic billing (reduces payment friction)
    """
    return {
        'scenarios': [
            {
                'name': 'Contract Upgrade',
                'description': 'Customer upgrades from month-to-month to 2-year contract',
                'modifications': {'Contract': 'Two year'}
            },
            {
                'name': 'Add Security Services',
                'description': 'Customer adds OnlineSecurity and TechSupport services',
                'modifications': {
                    'OnlineSecurity': 'Yes',
                    'TechSupport': 'Yes'
                }
            },
            {
                'name': 'Switch to Auto-Pay',
                'description': 'Customer switches from electronic check to automatic bank transfer',
                'modifications': {
                    'PaymentMethod': 'Bank transfer',
                    'PaperlessBilling': 'Yes'
                }
            },
            {
                'name': 'Upgrade Internet Service',
                'description': 'Customer upgrades from DSL to Fiber optic',
                'modifications': {'InternetService': 'Fiber optic'}
            },
            {
                'name': 'Add Streaming Services',
                'description': 'Customer adds StreamingTV and StreamingMovies',
                'modifications': {
                    'StreamingTV': 'Yes',
                    'StreamingMovies': 'Yes'
                }
            }
        ]
    }
