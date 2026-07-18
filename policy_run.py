"""
Run policy engine on pipeline_results.json entries.

Simple script to test the policy engine against benchmark results.
"""

import json
from pathlib import Path
from collections import Counter

from policy_engine import (
    PolicyEngine, 
    PolicyConfig, 
    ClassificationInput,
    PolicyTier
)


def load_pipeline_results(path: str = "benchmark_outputs/pipeline_results.json") -> list:
    """Load pipeline results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def result_to_classification(result: dict) -> ClassificationInput:
    """Convert a pipeline result dict to ClassificationInput."""
    
    # Check for direct identifiers in entity types
    direct_id_types = {"US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "CREDIT_CARD"}
    e_types = set(result.get("e_score_entity_types", []))
    has_direct_id = bool(e_types & direct_id_types)
    
    return ClassificationInput(
        # E-Score
        e_score=result.get("e_score", 0.0),
        e_score_entity_count=result.get("e_score_entity_count", 0),
        e_score_entity_types=result.get("e_score_entity_types", []),
        has_direct_identifier=has_direct_id or result.get("has_direct_identifier", False),
        e_score_masked_text=result.get("e_score_masked_text", ""),
        
        # Q-Score
        q_score=result.get("q_score", 0.0),
        q_score_expected_k=result.get("q_score_expected_k", 0),
        q_score_qi_count=result.get("q_score_qi_count", 0),
        has_rare_disease=result.get("q_score_has_rare_disease", False),
        has_rare_occupation=result.get("q_score_has_rare_occupation", False),
        q_score_masked_text=result.get("q_score_masked_text", ""),
        
        # C-Score
        c_score_risk_score=result.get("c_score_risk_score", 0.0),
        c_score_factors_detected=result.get("c_score_factors_detected", []),
        
        # Derived features
        has_person=result.get("has_person", False),
        has_location=result.get("has_location", False),
        has_datetime=result.get("has_datetime", False),
        has_high_risk_combo=result.get("has_high_risk_combo", False),
        
        # Original text
        original_text=result.get("original_text", "")
    )


def main():
    # Load results
    results_path = Path("benchmark_outputs/pipeline_results.json")
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        return
    
    results = load_pipeline_results(results_path)
    print(f"Loaded {len(results)} samples from {results_path}")
    
    # Initialize policy engine
    engine = PolicyEngine()
    
    # Track statistics
    tier_counts = Counter()
    routing_counts = Counter()
    
    # Confusion matrix: expected_critical vs policy tier
    expected_vs_tier = {
        (True, PolicyTier.RESTRICTED): 0,
        (True, PolicyTier.CONFIDENTIAL): 0,
        (True, PolicyTier.UNRESTRICTED): 0,
        (False, PolicyTier.RESTRICTED): 0,
        (False, PolicyTier.CONFIDENTIAL): 0,
        (False, PolicyTier.UNRESTRICTED): 0,
    }
    
    print("\n" + "=" * 80)
    print("POLICY DECISIONS")
    print("=" * 80)
    
    decisions_output = []
    
    for i, result in enumerate(results):
        sample_name = result.get("sample_name", f"Sample_{i}")
        expected_critical = result.get("expected_critical", False)
        
        # Convert to classification input
        classification = result_to_classification(result)
        
        # Evaluate policy
        decision = engine.evaluate(classification)
        
        # Track stats
        tier_counts[decision.tier] += 1
        routing_counts[decision.routing] += 1
        expected_vs_tier[(expected_critical, decision.tier)] += 1
        
        # Store for output
        decisions_output.append({
            "sample_name": sample_name,
            "expected_critical": expected_critical,
            "tier": decision.tier.value,
            "routing": decision.routing.value,
            "transformation": decision.transformation.value,
            "requires_attestation": decision.requires_attestation,
            "scores": {
                "e_score": decision.e_score,
                "q_score": decision.q_score,
                "c_score": decision.c_score,
                "combined": decision.combined_score
            },
            "reasons": decision.reasons
        })
        
        # Print summary for each sample
        tier_marker = {
            PolicyTier.RESTRICTED: "🔴 RESTRICTED",
            PolicyTier.CONFIDENTIAL: "🟡 CONFIDENTIAL", 
            PolicyTier.UNRESTRICTED: "🟢 UNRESTRICTED"
        }
        
        expected_marker = "⚠️  EXPECTED CRITICAL" if expected_critical else ""
        
        print(f"\n{sample_name}")
        print(f"  {tier_marker[decision.tier]} → {decision.routing.value} {expected_marker}")
        print(f"  Scores: E={decision.e_score:.2f}, Q={decision.q_score:.2f}, C={decision.c_score:.2f}, Sum={decision.combined_score:.2f}")
        print(f"  Reason: {decision.reasons[0]}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\nPolicy Tier Distribution:")
    for tier in PolicyTier:
        count = tier_counts[tier]
        pct = 100 * count / len(results) if results else 0
        bar = "█" * int(pct / 2)
        print(f"  {tier.value:15} {count:4} ({pct:5.1f}%) {bar}")
    
    print("\nRouting Distribution:")
    for routing, count in routing_counts.most_common():
        pct = 100 * count / len(results) if results else 0
        print(f"  {routing.value:15} {count:4} ({pct:5.1f}%)")
    
    print("\nExpected Critical vs Policy Tier:")
    print("  " + "-" * 50)
    print(f"  {'Expected':<20} {'RESTRICTED':>10} {'CONFIDENTIAL':>12} {'UNRESTRICTED':>12}")
    print("  " + "-" * 50)
    
    critical_row = [
        expected_vs_tier[(True, PolicyTier.RESTRICTED)],
        expected_vs_tier[(True, PolicyTier.CONFIDENTIAL)],
        expected_vs_tier[(True, PolicyTier.UNRESTRICTED)]
    ]
    not_critical_row = [
        expected_vs_tier[(False, PolicyTier.RESTRICTED)],
        expected_vs_tier[(False, PolicyTier.CONFIDENTIAL)],
        expected_vs_tier[(False, PolicyTier.UNRESTRICTED)]
    ]
    
    print(f"  {'Critical=True':<20} {critical_row[0]:>10} {critical_row[1]:>12} {critical_row[2]:>12}")
    print(f"  {'Critical=False':<20} {not_critical_row[0]:>10} {not_critical_row[1]:>12} {not_critical_row[2]:>12}")
    
    # Calculate some metrics
    # "Safe" = RESTRICTED or CONFIDENTIAL for critical samples
    critical_safe = critical_row[0] + critical_row[1]
    critical_total = sum(critical_row)
    critical_leak_rate = critical_row[2] / critical_total if critical_total > 0 else 0
    
    # "Efficient" = UNRESTRICTED for non-critical samples
    not_critical_total = sum(not_critical_row)
    not_critical_efficient = not_critical_row[2] / not_critical_total if not_critical_total > 0 else 0
    
    print("\nKey Metrics:")
    print(f"  Critical samples protected (RESTRICTED+CONFIDENTIAL): {critical_safe}/{critical_total} ({100*critical_safe/critical_total:.1f}%)" if critical_total > 0 else "  No critical samples")
    print(f"  Critical samples leaked (UNRESTRICTED): {critical_row[2]}/{critical_total} ({100*critical_leak_rate:.1f}%)" if critical_total > 0 else "")
    print(f"  Non-critical efficiency (UNRESTRICTED): {not_critical_row[2]}/{not_critical_total} ({100*not_critical_efficient:.1f}%)" if not_critical_total > 0 else "  No non-critical samples")
    
    # Save decisions to JSON
    output_path = Path("benchmark_outputs/policy_decisions.json")
    with open(output_path, "w") as f:
        json.dump(decisions_output, f, indent=2)
    print(f"\nSaved detailed decisions to {output_path}")


if __name__ == "__main__":
    main()