"""
Policy Engine for Software-Defined Confidential AI

Maps sensitivity classification scores to policy tiers and determines
routing, transformation, and handling decisions.

Based on benchmark analysis:
- E-Score: Explicit PII/PHI detection (Presidio)
- Q-Score: Quasi-identifier re-identification risk
- C-Score: Contextual/narrative sensitivity

Thresholds derived from benchmark correlation analysis and optimal k=10.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class PolicyTier(Enum):
    """The three policy tiers from the proposal."""
    UNRESTRICTED = "unrestricted"   # May use standard cloud APIs
    CONFIDENTIAL = "confidential"   # Requires masking + attested TEE endpoint
    RESTRICTED = "restricted"       # Local processing only, no network egress


class TransformationType(Enum):
    """What transformation to apply before transmission."""
    NONE = "none"                   # No transformation needed
    REVERSIBLE_MASK = "reversible"  # Mask with indexed placeholders, restore after
    IRREVERSIBLE_REDACT = "redact"  # Permanent removal, no restoration
    BLOCK = "block"                 # Do not transmit at all


class RoutingDecision(Enum):
    """Where to route the request."""
    LOCAL_MODEL = "local"           # Process with local model only
    TEE_ENDPOINT = "tee"            # Route to attested confidential endpoint
    CLOUD_API = "cloud"             # Route to standard cloud LLM API
    BLOCKED = "blocked"             # Request blocked entirely


@dataclass
class PolicyConfig:
    """
    Configurable thresholds for policy decisions.
    
    Defaults derived from benchmark analysis:
    - C-Score optimal threshold: 0.12 (from precision-recall analysis)
    - E-Score threshold: 0.5 (high explicit PII presence)
    - Q-Score threshold: Based on k=10 anonymity
    - Direct identifiers (SSN, etc.) always trigger RESTRICTED
    """
    # Score thresholds
    e_score_high: float = 0.5           # Above this = high explicit PII
    e_score_moderate: float = 0.3       # Above this = moderate PII
    
    q_score_high: float = 0.5           # Above this = high re-id risk
    q_score_moderate: float = 0.2       # Above this = moderate re-id risk
    
    c_score_sensitive: float = 0.12     # From your optimal threshold analysis
    c_score_high: float = 0.3           # High contextual sensitivity
    
    # Combined thresholds
    combined_score_restricted: float = 1.2  # E+Q+C sum triggers restricted
    combined_score_confidential: float = 0.6
    
    # k-anonymity threshold (from your optimal k analysis)
    k_anonymity_threshold: int = 10
    
    # Feature-based overrides (from your boolean feature analysis)
    direct_id_always_restricted: bool = True    # SSN, passport, etc.
    high_risk_combo_elevates: bool = True       # PERSON+LOCATION+DATETIME


@dataclass
class ClassificationInput:
    """Input from the E-Score, Q-Score, and C-Score pipeline."""
    # E-Score results
    e_score: float
    e_score_entity_count: int
    e_score_entity_types: List[str]
    has_direct_identifier: bool  # SSN, passport, driver's license, etc.
    e_score_masked_text: str
    
    # Q-Score results
    q_score: float
    q_score_expected_k: float
    q_score_qi_count: int
    has_rare_disease: bool
    has_rare_occupation: bool
    q_score_masked_text: str
    
    # C-Score results
    c_score_risk_score: float
    c_score_factors_detected: List[str]
    
    # Derived features
    has_person: bool = False
    has_location: bool = False
    has_datetime: bool = False
    has_high_risk_combo: bool = False
    
    # Original text (for masking operations)
    original_text: str = ""


@dataclass
class PolicyDecision:
    """The output of the policy engine."""
    tier: PolicyTier
    transformation: TransformationType
    routing: RoutingDecision
    
    # Explanations for audit
    reasons: List[str] = field(default_factory=list)
    
    # Scores that led to decision
    e_score: float = 0.0
    q_score: float = 0.0
    c_score: float = 0.0
    combined_score: float = 0.0
    
    # For reversible masking
    requires_reconstruction: bool = False
    masked_text: Optional[str] = None
    
    # Attestation requirements
    requires_attestation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "transformation": self.transformation.value,
            "routing": self.routing.value,
            "reasons": self.reasons,
            "scores": {
                "e_score": self.e_score,
                "q_score": self.q_score,
                "c_score": self.c_score,
                "combined": self.combined_score
            },
            "requires_reconstruction": self.requires_reconstruction,
            "requires_attestation": self.requires_attestation
        }


class PolicyEngine:
    """
    Determines policy tier, transformation, and routing based on
    sensitivity classification scores.
    
    Decision hierarchy (highest priority first):
    1. Direct identifiers → RESTRICTED
    2. High C-Score (contextual sensitivity) → RESTRICTED or CONFIDENTIAL
    3. High E-Score + Q-Score combination → CONFIDENTIAL
    4. Moderate scores with risk factors → CONFIDENTIAL
    5. Low scores → UNRESTRICTED
    """
    
    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
    
    def evaluate(self, classification: ClassificationInput) -> PolicyDecision:
        """
        Evaluate classification scores and return a policy decision.
        """
        reasons = []
        
        # Calculate combined score
        combined = (
            classification.e_score + 
            classification.q_score + 
            classification.c_score_risk_score
        )
        
        # =====================================================================
        # RULE 1: Direct identifiers → RESTRICTED (always)
        # =====================================================================
        if self.config.direct_id_always_restricted and classification.has_direct_identifier:
            reasons.append("Direct identifier detected (SSN, passport, etc.)")
            return PolicyDecision(
                tier=PolicyTier.RESTRICTED,
                transformation=TransformationType.BLOCK,
                routing=RoutingDecision.LOCAL_MODEL,
                reasons=reasons,
                e_score=classification.e_score,
                q_score=classification.q_score,
                c_score=classification.c_score_risk_score,
                combined_score=combined,
                requires_reconstruction=False,
                masked_text=None,
                requires_attestation=False
            )
        
        # =====================================================================
        # RULE 2: Very high combined score → RESTRICTED
        # =====================================================================
        if combined >= self.config.combined_score_restricted:
            reasons.append(f"Combined score {combined:.2f} >= {self.config.combined_score_restricted}")
            return PolicyDecision(
                tier=PolicyTier.RESTRICTED,
                transformation=TransformationType.BLOCK,
                routing=RoutingDecision.LOCAL_MODEL,
                reasons=reasons,
                e_score=classification.e_score,
                q_score=classification.q_score,
                c_score=classification.c_score_risk_score,
                combined_score=combined,
                requires_reconstruction=False,
                masked_text=None,
                requires_attestation=False
            )
        
        # =====================================================================
        # RULE 3: High C-Score (contextual sensitivity) → RESTRICTED
        # Contextual sensitivity often can't be masked away
        # =====================================================================
        if classification.c_score_risk_score >= self.config.c_score_high:
            reasons.append(f"High contextual sensitivity: C-Score {classification.c_score_risk_score:.2f}")
            if classification.c_score_factors_detected:
                reasons.append(f"Factors: {', '.join(classification.c_score_factors_detected)}")
            return PolicyDecision(
                tier=PolicyTier.RESTRICTED,
                transformation=TransformationType.BLOCK,
                routing=RoutingDecision.LOCAL_MODEL,
                reasons=reasons,
                e_score=classification.e_score,
                q_score=classification.q_score,
                c_score=classification.c_score_risk_score,
                combined_score=combined,
                requires_reconstruction=False,
                masked_text=None,
                requires_attestation=False
            )
        
        # =====================================================================
        # RULE 4: High E-Score OR rare QI → CONFIDENTIAL with masking
        # =====================================================================
        if classification.e_score >= self.config.e_score_high:
            reasons.append(f"High explicit PII: E-Score {classification.e_score:.2f}")
            return self._make_confidential_decision(classification, reasons, combined)
        
        if classification.has_rare_disease or classification.has_rare_occupation:
            reasons.append("Rare quasi-identifier detected (disease or occupation)")
            return self._make_confidential_decision(classification, reasons, combined)
        
        if classification.q_score >= self.config.q_score_high:
            reasons.append(f"High re-identification risk: Q-Score {classification.q_score:.2f}")
            return self._make_confidential_decision(classification, reasons, combined)
        
        # =====================================================================
        # RULE 5: Moderate sensitivity with risk combos → CONFIDENTIAL
        # =====================================================================
        if (self.config.high_risk_combo_elevates and 
            classification.has_high_risk_combo and
            combined >= self.config.combined_score_confidential):
            reasons.append("High-risk entity combination (PERSON+LOCATION+DATETIME)")
            reasons.append(f"Combined score {combined:.2f} with risk factors")
            return self._make_confidential_decision(classification, reasons, combined)
        
        # =====================================================================
        # RULE 6: Moderate C-Score → CONFIDENTIAL with masking
        # =====================================================================
        if classification.c_score_risk_score >= self.config.c_score_sensitive:
            reasons.append(f"Moderate contextual sensitivity: C-Score {classification.c_score_risk_score:.2f}")
            return self._make_confidential_decision(classification, reasons, combined)
        
        # =====================================================================
        # RULE 7: Moderate E-Score + any Q-Score → CONFIDENTIAL
        # =====================================================================
        if (classification.e_score >= self.config.e_score_moderate and 
            classification.q_score > 0):
            reasons.append(f"Moderate PII with quasi-identifiers present")
            return self._make_confidential_decision(classification, reasons, combined)
        
        # =====================================================================
        # DEFAULT: Low sensitivity → UNRESTRICTED
        # =====================================================================
        reasons.append("No significant sensitivity indicators detected")
        reasons.append(f"Scores: E={classification.e_score:.2f}, Q={classification.q_score:.2f}, C={classification.c_score_risk_score:.2f}")
        
        return PolicyDecision(
            tier=PolicyTier.UNRESTRICTED,
            transformation=TransformationType.NONE,
            routing=RoutingDecision.CLOUD_API,
            reasons=reasons,
            e_score=classification.e_score,
            q_score=classification.q_score,
            c_score=classification.c_score_risk_score,
            combined_score=combined,
            requires_reconstruction=False,
            masked_text=None,
            requires_attestation=False
        )
    
    def _make_confidential_decision(
        self, 
        classification: ClassificationInput, 
        reasons: List[str],
        combined: float
    ) -> PolicyDecision:
        """Helper to create a CONFIDENTIAL tier decision with masking."""
        
        # Use the Q-Score masked text (which was run on E-Score masked text)
        # This represents the fully masked version
        masked_text = classification.q_score_masked_text or classification.e_score_masked_text
        
        return PolicyDecision(
            tier=PolicyTier.CONFIDENTIAL,
            transformation=TransformationType.REVERSIBLE_MASK,
            routing=RoutingDecision.TEE_ENDPOINT,
            reasons=reasons,
            e_score=classification.e_score,
            q_score=classification.q_score,
            c_score=classification.c_score_risk_score,
            combined_score=combined,
            requires_reconstruction=True,
            masked_text=masked_text,
            requires_attestation=True
        )


# =============================================================================
# Convenience function for simple usage
# =============================================================================

def evaluate_policy(
    e_score: float,
    q_score: float,
    c_score: float,
    has_direct_identifier: bool = False,
    has_high_risk_combo: bool = False,
    e_score_entity_types: Optional[List[str]] = None,
    c_score_factors: Optional[List[str]] = None,
    masked_text: str = "",
    original_text: str = "",
    config: Optional[PolicyConfig] = None
) -> PolicyDecision:
    """
    Simple function interface for policy evaluation.
    
    Example:
        decision = evaluate_policy(
            e_score=0.65,
            q_score=0.3,
            c_score=0.15,
            has_direct_identifier=False
        )
        print(decision.tier)  # PolicyTier.CONFIDENTIAL
    """
    classification = ClassificationInput(
        e_score=e_score,
        e_score_entity_count=0,
        e_score_entity_types=e_score_entity_types or [],
        has_direct_identifier=has_direct_identifier,
        e_score_masked_text=masked_text,
        q_score=q_score,
        q_score_expected_k=0,
        q_score_qi_count=0,
        has_rare_disease=False,
        has_rare_occupation=False,
        q_score_masked_text=masked_text,
        c_score_risk_score=c_score,
        c_score_factors_detected=c_score_factors or [],
        has_high_risk_combo=has_high_risk_combo,
        original_text=original_text
    )
    
    engine = PolicyEngine(config)
    return engine.evaluate(classification)


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("POLICY ENGINE TEST CASES")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "Low sensitivity - general health question",
            "e_score": 0.1,
            "q_score": 0.0,
            "c_score": 0.05,
            "has_direct_identifier": False,
        },
        {
            "name": "Moderate PII - masked patient name",
            "e_score": 0.45,
            "q_score": 0.15,
            "c_score": 0.08,
            "has_direct_identifier": False,
        },
        {
            "name": "High PII with SSN",
            "e_score": 0.75,
            "q_score": 0.3,
            "c_score": 0.1,
            "has_direct_identifier": True,
        },
        {
            "name": "High contextual sensitivity - narrative uniqueness",
            "e_score": 0.2,
            "q_score": 0.1,
            "c_score": 0.45,
            "has_direct_identifier": False,
            "c_score_factors": ["NARRATIVE_UNIQUENESS", "INFERENTIAL_DISCLOSURE"]
        },
        {
            "name": "Moderate with high-risk combo",
            "e_score": 0.35,
            "q_score": 0.25,
            "c_score": 0.12,
            "has_direct_identifier": False,
            "has_high_risk_combo": True,
        },
        {
            "name": "Borderline - just above C-score threshold",
            "e_score": 0.15,
            "q_score": 0.05,
            "c_score": 0.13,
            "has_direct_identifier": False,
        },
    ]
    
    for tc in test_cases:
        name = tc.pop("name")
        decision = evaluate_policy(**tc)
        
        print(f"\n{name}")
        print("-" * 50)
        print(f"  Tier: {decision.tier.value.upper()}")
        print(f"  Routing: {decision.routing.value}")
        print(f"  Transform: {decision.transformation.value}")
        print(f"  Requires Attestation: {decision.requires_attestation}")
        print(f"  Requires Reconstruction: {decision.requires_reconstruction}")
        print(f"  Reasons:")
        for reason in decision.reasons:
            print(f"    - {reason}")