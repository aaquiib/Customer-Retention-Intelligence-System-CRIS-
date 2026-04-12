"""Business rules engine for retention action and priority scoring."""

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.utils import load_json

logger = logging.getLogger(__name__)


class RetentionActionDecider:
    """
    Config-driven business rules engine for retention actions and priority scoring.

    Decides what action to recommend for each customer based on:
    - Segment (loyalty, value tier)
    - Churn probability (risk)
    - Customer lifetime value / tenure
    - Contract type and service usage

    All thresholds and action mappings are defined in config/business_rules.yaml
    """

    def __init__(self, rules_config_path: str = "config/business_rules.yaml"):
        """
        Initialize action decider with business rules from YAML.

        Args:
            rules_config_path: Path to business_rules.yaml

        Raises:
            FileNotFoundError: If business_rules.yaml not found
        """
        try:
            # Note: Assuming business_rules.yaml uses JSON format for now
            # Will be created in next step
            self.rules = load_json(rules_config_path.replace('.yaml', '.json'))
            self._validate_rules()
            logger.info(f"✓ Loaded business rules from {rules_config_path}")
        except FileNotFoundError:
            logger.warning(f"Business rules file not found at {rules_config_path}. Using defaults.")
            self.rules = self._get_default_rules()

    def _validate_rules(self) -> None:
        """Validate that rules have required structure."""
        required_keys = ['actions', 'priority_weights', 'segment_risk_scores']
        for key in required_keys:
            if key not in self.rules:
                raise ValueError(f"Missing required key in business rules: {key}")

    def _get_default_rules(self) -> Dict[str, Any]:
        """Return default business rules if config not found."""
        return {
            'actions': {
                'do_nothing': {
                    'label': 'Monitor',
                    'description': 'Low risk - routine monitoring'
                },
                'loyalty_program': {
                    'label': 'Loyalty Program',
                    'description': 'Offer loyalty rewards and benefits'
                },
                'discount_offer': {
                    'label': 'Discount Offer',
                    'description': 'Offer service discounts or bundle deals'
                },
                'retention_call': {
                    'label': 'Retention Call',
                    'description': 'Direct outreach with personalized retention offer'
                },
                'vip_support': {
                    'label': 'VIP Support',
                    'description': 'Upgrade to premium support with dedicated account manager'
                },
                'early_exit_waiver': {
                    'label': 'Early Exit Waiver',
                    'description': 'Waive early termination fees if customer still leaves'
                }
            },
            'priority_weights': {
                'churn_probability': 0.40,
                'segment_risk': 0.30,
                'customer_value': 0.30
            },
            'segment_risk_scores': {
                'Loyal High-Value': 0.1,
                'Low Engagement': 0.6,
                'Stable Mid-Value': 0.3,
                'At risk High-value': 0.8
            },
            'thresholds': {
                'high_value_tenure': 36,
                'high_value_monthly_charges': 75.0,
                'low_value_monthly_charges': 30.0,
                'high_churn_prob': 0.6,
                'medium_churn_prob': 0.4
            }
        }

    def decide_action(
        self,
        segment: int,
        segment_label: str,
        churn_probability: float,
        customer_features: Dict[str, Any],
        segment_confidence: float = 1.0
    ) -> Tuple[str, str, float, str]:
        """
        Decide retention action based on customer profile.

        Args:
            segment: Segment ID (0-3)
            segment_label: Segment label string
            churn_probability: Predicted churn probability (0.0-1.0)
            customer_features: Original customer features dict
            segment_confidence: Confidence score of segment assignment (0.0-1.0)

        Returns:
            Tuple of (action_key, action_label, priority_score, reason)
            where:
                - action_key: machine-readable action code ('loyalty_program', 'discount_offer', etc.)
                - action_label: human-readable label
                - priority_score: 0.0-1.0, where 1.0 = highest priority
                - reason: Explanation of why this action was chosen
        """
        # Compute priority score as weighted combination
        priority_score = self._compute_priority_score(
            churn_probability,
            segment_label,
            customer_features
        )

        # Determine customer value tier
        value_tier = self._evaluate_customer_value(customer_features)

        # Determine action based on segment, churn risk, and value
        action_key = self._select_action(
            segment_label,
            churn_probability,
            value_tier,
            customer_features
        )

        # Get action label and build reason
        action_info = self.rules['actions'].get(action_key, {})
        action_label = action_info.get('label', action_key)
        reason = self._generate_reason(
            segment_label,
            churn_probability,
            value_tier,
            action_key
        )

        logger.debug(f"Action decision: {action_key} | Priority: {priority_score:.3f} | {reason}")

        return action_key, action_label, priority_score, reason

    def _compute_priority_score(
        self,
        churn_probability: float,
        segment_label: str,
        customer_features: Dict[str, Any]
    ) -> float:
        """
        Compute priority score (0.0-1.0) as weighted combination.

        Higher score = higher priority for intervention.
        """
        weights = self.rules['priority_weights']

        # Churn probability component
        churn_component = churn_probability * weights['churn_probability']

        # Segment risk component
        segment_risk_scores = self.rules['segment_risk_scores']
        segment_risk = segment_risk_scores.get(segment_label, 0.5)
        risk_component = segment_risk * weights['segment_risk']

        # Customer value component (inverse - lower value = higher priority)
        value_tier = self._evaluate_customer_value(customer_features)
        value_component = (1.0 - self._value_tier_to_score(value_tier)) * weights['customer_value']

        priority_score = churn_component + risk_component + value_component
        return min(max(priority_score, 0.0), 1.0)

    def _evaluate_customer_value(self, customer_features: Dict[str, Any]) -> str:
        """Classify customer as 'high', 'medium', or 'low' value."""
        thresholds = self.rules['thresholds']
        tenure = customer_features.get('tenure', 0)
        monthly_charges = customer_features.get('MonthlyCharges', 0.0)

        # High value: long tenure AND high monthly spend
        if (tenure >= thresholds.get('high_value_tenure_months', 36) and 
            monthly_charges >= thresholds.get('high_value_monthly_charges', 75.0)):
            return 'high'
        
        # Low value: low monthly spend
        elif monthly_charges < thresholds.get('low_value_monthly_charges', 30.0):
            return 'low'
        
        else:
            return 'medium'

    def _value_tier_to_score(self, value_tier: str) -> float:
        """Convert value tier to score (0=low value, 1=high value)."""
        return {'low': 0.2, 'medium': 0.5, 'high': 0.9}.get(value_tier, 0.5)

    def _select_action(
        self,
        segment_label: str,
        churn_probability: float,
        value_tier: str,
        customer_features: Dict[str, Any]
    ) -> str:
        """
        Select appropriate action based on decision tree logic.

        Decision logic:
        - High churn prob + High value: vip_support or early_exit_waiver
        - High churn prob + Medium value: retention_call
        - Medium churn prob + Any value: discount_offer
        - Low churn prob + High value: loyalty_program
        - Low churn prob + Any: do_nothing
        """
        thresholds = self.rules['thresholds']

        # High churn probability
        if churn_probability >= thresholds.get('high_churn_prob', 0.60):
            if value_tier == 'high':
                # Are they in a flexible contract? Offer early exit waiver
                contract = customer_features.get('Contract', 'Month-to-month')
                if 'Month' in contract:
                    return 'early_exit_waiver'
                else:
                    return 'vip_support'
            elif value_tier == 'medium':
                return 'retention_call'
            else:
                # Low value + high churn: let them go, but offer discount to retain
                return 'discount_offer'

        # Medium churn probability
        elif churn_probability >= thresholds.get('medium_churn_prob', 0.40):
            if value_tier == 'high':
                return 'discount_offer'
            elif value_tier == 'medium':
                return 'retention_call'
            else:
                return 'do_nothing'

        # Low churn probability
        else:
            if value_tier == 'high':
                return 'loyalty_program'
            else:
                return 'do_nothing'

    def _generate_reason(
        self,
        segment_label: str,
        churn_probability: float,
        value_tier: str,
        action_key: str
    ) -> str:
        """Generate human-readable explanation for the action decision."""
        reasons = []

        # Segment insight
        reasons.append(f"Segment: {segment_label}")

        # Churn risk insight
        if churn_probability >= 0.6:
            reasons.append(f"High churn risk ({churn_probability:.1%})")
        elif churn_probability >= 0.4:
            reasons.append(f"Medium churn risk ({churn_probability:.1%})")
        else:
            reasons.append(f"Low churn risk ({churn_probability:.1%})")

        # Value insight
        reasons.append(f"Customer value: {value_tier.capitalize()}")

        return " | ".join(reasons)


def get_action_decider(rules_config_path: str = "config/business_rules.json") -> RetentionActionDecider:
    """Factory function to create RetentionActionDecider singleton."""
    return RetentionActionDecider(rules_config_path)
