"""
Test suite for causal discovery and counterfactual reasoning.

These queries test the intervention and counterfactual logic across:
1. Cross-sector causal pathways
2. Policy interventions with geographic variation
3. Natural experiments (temporal and spatial)
4. Counterfactual scenarios with historical precedents
"""

import logging
from reasoning.query_decomposer import QueryDecomposer
from reasoning.metadata_analyzer import MetadataAnalyzer
from reasoning.causal_discovery import CausalDiscovery
from reasoning.counterfactual_engine import CounterfactualEngine
from reasoning.insights_generator import InsightsGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize reasoning modules
decomposer = QueryDecomposer()
analyzer = MetadataAnalyzer()
causal = CausalDiscovery()
counterfactual = CounterfactualEngine()
insights = InsightsGenerator()

# ============================================================================
# Test Suite 1: Cross-Sector Causal Pathways
# ============================================================================

CAUSAL_PATHWAY_TESTS = [
    {
        "name": "Education → Income → Health",
        "question": "What is the causal pathway from education investment to health outcomes?",
        "description": "Tests if system identifies multi-hop causal chains across sectors",
        "expected_sectors": ["Education", "Economy", "Health"],
        "expected_methods": ["DiD", "IV", "matching"],
    },
    {
        "name": "Infrastructure → Economic Growth",
        "question": "Can we measure the causal effect of road infrastructure on economic growth?",
        "description": "Tests identification of policy interventions with geographic variation",
        "expected_sectors": ["Transport", "Economy"],
        "expected_methods": ["DiD", "RD", "synthetic_control"],
    },
    {
        "name": "Water Access → Health + Agriculture",
        "question": "How does water resource availability affect both agricultural productivity and health?",
        "description": "Tests multi-outcome causal analysis",
        "expected_sectors": ["Water Resources", "Agriculture", "Health"],
        "expected_methods": ["matching", "synthetic_control"],
    },
    {
        "name": "Finance & Development",
        "question": "What sectors show strongest correlation with financial investment patterns?",
        "description": "Tests identification of sectors with finance linkages",
        "expected_sectors": ["Finance", "Energy", "Agriculture", "Education"],
        "expected_methods": ["IV", "matching"],
    },
]

# ============================================================================
# Test Suite 2: Policy Interventions (Difference-in-Differences)
# ============================================================================

INTERVENTION_TESTS = [
    {
        "name": "Conditional Cash Transfer (CCT) on Education",
        "question": "What would be the effect of expanding conditional cash transfers to all districts?",
        "description": "Tests DiD-style intervention logic with treatment/control groups",
        "treatment_variable": "education_spending",
        "outcome": "enrollment_rates",
        "geographic_variation": True,  # Some states have CCT, others don't
        "temporal": "2010-2024",
    },
    {
        "name": "Electrification Program Impact",
        "question": "If rural electrification was expanded to all unelectrified districts, what would happen to education and health outcomes?",
        "description": "Tests natural experiment with staggered rollout",
        "treatment_variable": "electrification_rate",
        "outcome": ["school_enrollment", "health_center_visits"],
        "geographic_variation": True,
        "temporal": "2015-2024",
    },
    {
        "name": "Agricultural Subsidy Effectiveness",
        "question": "What is the causal effect of crop insurance on agricultural productivity?",
        "description": "Tests policy intervention with some states adopting earlier than others",
        "treatment_variable": "crop_insurance_coverage",
        "outcome": "agricultural_yield",
        "geographic_variation": True,
        "temporal": "2010-2024",
    },
    {
        "name": "Health Center Availability",
        "question": "Do districts with more health centers have better health outcomes?",
        "description": "Tests whether infrastructure investment causes outcome improvements",
        "treatment_variable": "health_center_density",
        "outcome": ["infant_mortality", "maternal_mortality", "disease_prevalence"],
        "geographic_variation": True,
        "temporal": "2015-2024",
    },
]

# ============================================================================
# Test Suite 3: Regression Discontinuity (Policy Cutoffs)
# ============================================================================

DISCONTINUITY_TESTS = [
    {
        "name": "Population-Based Program Eligibility",
        "question": "Districts above 1M population get preferential funding. Does this threshold cause a jump in development?",
        "description": "Tests RD with population threshold",
        "running_variable": "district_population",
        "cutoff": 1000000,
        "outcome": "development_index",
    },
    {
        "name": "Poverty-Line Benefit Eligibility",
        "question": "Do districts just below the poverty line threshold receive disproportionate benefits?",
        "description": "Tests RD with poverty metric",
        "running_variable": "poverty_rate",
        "cutoff": 0.30,  # 30% poverty threshold
        "outcome": "public_spending",
    },
]

# ============================================================================
# Test Suite 4: Counterfactual Scenarios
# ============================================================================

COUNTERFACTUAL_TESTS = [
    {
        "name": "What if all states adopted best-performer's policies?",
        "question": "If all states adopted the education spending levels of the top-performing state, what would be the projected enrollment gain?",
        "scenario": "Policy harmonization",
        "treatment": "education_budget_increase",
        "magnitude": "Match top 10%",
        "comparison_strategy": "cross-state_comparison",
    },
    {
        "name": "Synthetic Control: Tamil Nadu's Health Model",
        "question": "What would have happened to health outcomes in other states if they had adopted Tamil Nadu's health system design from 2015 onwards?",
        "scenario": "Health system reform",
        "treatment": "health_system_modernization",
        "magnitude": "Full adoption",
        "comparison_strategy": "synthetic_control",
        "treated_unit": "Tamil Nadu",
    },
    {
        "name": "Staggered Rollout Analysis",
        "question": "States rolled out digital literacy programs at different times (2015-2020). Can we estimate the effect by comparing early vs late adopters?",
        "scenario": "Technology adoption timing",
        "treatment": "digital_literacy_program",
        "magnitude": "Full coverage",
        "comparison_strategy": "staggered_did",
        "timing_variation": True,
    },
    {
        "name": "Inverse Counterfactual",
        "question": "What would have happened to transport accidents if helmet and seatbelt laws had NOT been introduced?",
        "scenario": "Policy reversal",
        "treatment": "traffic_safety_law",
        "magnitude": "Complete removal",
        "comparison_strategy": "pre_post_comparison",
    },
    {
        "name": "Spillover Effects",
        "question": "When state A improved water infrastructure, did neighboring states B and C also benefit from spillovers?",
        "scenario": "Spatial spillovers",
        "treatment": "infrastructure_investment",
        "magnitude": "20% increase",
        "comparison_strategy": "spatial_analysis",
        "spatial_neighbors": True,
    },
]

# ============================================================================
# Test Suite 5: Causal Identification Challenges
# ============================================================================

IDENTIFICATION_CHALLENGE_TESTS = [
    {
        "name": "Reverse Causality Problem",
        "question": "Does economic growth cause more health spending, or does health spending cause growth?",
        "challenge": "reverse_causality",
        "needs_instrument": True,
    },
    {
        "name": "Omitted Variable Bias",
        "question": "Does education spending cause better health outcomes, or do wealthy districts do both?",
        "challenge": "confounding",
        "needs_control_variables": True,
    },
    {
        "name": "Selection Bias",
        "question": "Districts that voluntarily joined the water conservation program show better outcomes. Is this the program or self-selection?",
        "challenge": "selection_bias",
        "needs_matching": True,
    },
    {
        "name": "Measurement Error",
        "question": "Can we trust health outcome data from poorly-resourced districts?",
        "challenge": "measurement_error",
        "needs_validation": True,
    },
]

# ============================================================================
# Test Suite 6: Complex Multi-Intervention Scenarios
# ============================================================================

COMPLEX_INTERVENTION_TESTS = [
    {
        "name": "NREGA + Irrigation + Seeds",
        "question": "What is the combined causal effect of implementing NREGA, irrigation improvement, AND distributing improved seeds together?",
        "interventions": [
            "rural_employment_guarantee",
            "irrigation_expansion",
            "seed_distribution",
        ],
        "outcome": "agricultural_income",
        "question_type": "synergy_analysis",
    },
    {
        "name": "Education-Health-Economic Trifecta",
        "question": "Trace the full pathway: female education → workforce participation → household income → child health",
        "interventions": ["girls_education_grant"],
        "outcomes": ["female_workforce", "household_income", "child_mortality"],
        "question_type": "mediation_analysis",
        "pathways": 3,
    },
    {
        "name": "Policy Phase-Out Effect",
        "question": "When subsidies were reduced, did beneficiaries substitute to private alternatives or go without?",
        "interventions": ["subsidy_reduction"],
        "outcomes": ["private_consumption", "program_enrollment"],
        "question_type": "substitution_analysis",
    },
]

# ============================================================================
# Test Suite 7: Data Availability and Granularity Matching
# ============================================================================

GRANULARITY_TESTS = [
    {
        "name": "Mismatch: State data meets District question",
        "question": "Can we estimate district-level effects when we only have state-level health data?",
        "treatment_granularity": "state",
        "outcome_granularity": "district",
        "issue": "aggregation_bias",
    },
    {
        "name": "Temporal Mismatch",
        "question": "Can we link education data (annual) to health outcomes (monthly)?",
        "treatment_temporal": "annual",
        "outcome_temporal": "monthly",
        "issue": "temporal_aggregation",
    },
    {
        "name": "Sparse Coverage",
        "question": "Only 3 states have detailed village-level data. Can we extrapolate to national level?",
        "coverage": "3_out_of_28_states",
        "granularity": "village",
        "issue": "external_validity",
    },
]

# ============================================================================
# Test Functions
# ============================================================================


def test_causal_pathway(test_case):
    """Test causal pathway discovery."""
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST: {test_case['name']}")
    logger.info(f"{'='*70}")
    logger.info(f"Question: {test_case['question']}")
    logger.info(f"Description: {test_case.get('description', test_case.get('question_type', 'N/A'))}")

    # Step 1: Decompose
    decomposed = decomposer.decompose(test_case['question'])
    logger.info(f"\n→ Decomposed intent: {decomposed.intent}")
    logger.info(f"  Sectors: {decomposed.entities.get('sectors', [])}")
    logger.info(f"  Metrics: {decomposed.entities.get('metrics', [])}")
    logger.info(f"  Granularity: {decomposed.required_granularity}")
    logger.info(f"  Sub-questions: {decomposed.sub_questions}")

    # Step 2: Analyze feasibility
    report = analyzer.analyze(decomposed)
    logger.info(f"\n→ Coverage score: {report.coverage_score:.2f}")
    logger.info(f"  Found {len(report.found_datasets)} datasets")
    logger.info(f"  Linkable pairs: {len(report.linkable_pairs)}")
    logger.info(f"  Gaps: {report.gaps}")
    logger.info(f"  Granularity matched: {report.granularity_matched}")

    # Step 3: Discover causal pathways
    plan = causal.discover(test_case['question'], report)
    logger.info(f"\n→ Causal Analysis:")
    logger.info(f"  Pathways found: {len(plan.pathways)}")
    for pw in plan.pathways:
        logger.info(f"    - {' → '.join(pw.chain)} ({pw.strength})")

    logger.info(f"  Methods suggested: {len(plan.methods)}")
    for method in plan.methods:
        logger.info(f"    - {method.name} ({method.abbreviation}): feasible={method.feasible}")

    logger.info(f"  Overall feasibility: {plan.overall_feasibility}")

    # Step 4: Generate insights
    insight_report = insights.generate(test_case['question'], report, causal=plan)
    logger.info(f"\n→ Insights:")
    logger.info(f"  Summary: {insight_report.executive_summary}")
    logger.info(f"  Data gaps: {insight_report.data_gaps}")

    # Validate
    assert len(plan.pathways) > 0, "No causal pathways identified"
    assert plan.overall_feasibility in ("feasible", "partially_feasible", "infeasible")
    logger.info(f"✅ PASS")
    return plan


def test_intervention(test_case):
    """Test intervention/causal effect logic."""
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST: {test_case['name']}")
    logger.info(f"{'='*70}")
    logger.info(f"Question: {test_case['question']}")
    logger.info(f"Treatment: {test_case['treatment_variable']}")
    logger.info(f"Outcome: {test_case['outcome']}")

    # Step 1: Decompose
    decomposed = decomposer.decompose(test_case['question'])
    logger.info(f"\n→ Decomposed: intent={decomposed.intent}")

    # Step 2: Analyze
    report = analyzer.analyze(decomposed)
    logger.info(f"→ Coverage score: {report.coverage_score:.2f} | {len(report.found_datasets)} datasets")

    # Step 3: Causal discovery
    plan = causal.discover(test_case['question'], report)
    logger.info(f"→ Methods: {[m.abbreviation for m in plan.methods if m.feasible]}")
    logger.info(f"→ Identification checks:")
    for check, value in plan.identification_checks.items():
        logger.info(f"    {check}: {value}")

    # Validate: expect at least one feasible method
    feasible_methods = [m for m in plan.methods if m.feasible]
    assert len(feasible_methods) > 0, "No feasible methods found for intervention"
    logger.info(f"✅ PASS")
    return plan


def test_counterfactual(test_case):
    """Test counterfactual scenario analysis."""
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST: {test_case['name']}")
    logger.info(f"{'='*70}")
    logger.info(f"Question: {test_case['question']}")
    logger.info(f"Scenario: {test_case['scenario']}")
    logger.info(f"Strategy: {test_case['comparison_strategy']}")

    # Step 1: Decompose
    decomposed = decomposer.decompose(test_case['question'])
    logger.info(f"\n→ Decomposed: {decomposed.intent}")

    # Step 2: Analyze
    report = analyzer.analyze(decomposed)
    logger.info(f"→ Found {len(report.found_datasets)} datasets")

    # Step 3: Plan counterfactual
    cf_plan = counterfactual.plan(test_case['question'], report)
    logger.info(f"\n→ Counterfactual Plan:")
    logger.info(f"  Treatment variable: {cf_plan.treatment_variable}")
    logger.info(f"  Outcome variables: {cf_plan.outcome_variables}")
    logger.info(f"  Comparison strategy: {cf_plan.comparison_strategy}")
    logger.info(f"  Historical precedents: {len(cf_plan.historical_precedents)}")
    for prec in cf_plan.historical_precedents:
        logger.info(f"    - {prec.description} (relevance: {prec.relevance})")

    logger.info(f"  Assumptions: {len(cf_plan.assumptions)}")
    for assumption in cf_plan.assumptions:
        logger.info(f"    - {assumption.statement} (testable: {assumption.testable})")

    logger.info(f"  Feasibility score: {cf_plan.feasibility_score:.2f}")
    logger.info(f"  Notes: {cf_plan.feasibility_notes}")

    # Validate
    assert cf_plan.feasibility_score >= 0.0 and cf_plan.feasibility_score <= 1.0
    logger.info(f"✅ PASS")
    return cf_plan


def test_identification_challenge(test_case):
    """Test whether system recognizes causal identification challenges."""
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST: {test_case['name']}")
    logger.info(f"{'='*70}")
    logger.info(f"Question: {test_case['question']}")
    logger.info(f"Challenge: {test_case['challenge']}")

    # Step 1: Decompose
    decomposed = decomposer.decompose(test_case['question'])

    # Step 2: Analyze
    report = analyzer.analyze(decomposed)

    # Step 3: Discover causal issues
    plan = causal.discover(test_case['question'], report)
    logger.info(f"\n→ Feasibility: {plan.overall_feasibility}")
    logger.info(f"→ Notes: {plan.notes}")

    # Step 4: Generate insights — data_gaps captures limitations
    insight_report = insights.generate(test_case['question'], report, causal=plan)
    logger.info(f"\n→ Identified Limitations / Data Gaps:")
    for gap in insight_report.data_gaps:
        logger.info(f"    - {gap}")
    logger.info(f"→ Methodology notes: {insight_report.methodology_notes}")

    logger.info(f"✅ PASS")
    return plan


# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("DHURANDHAAR CAUSAL REASONING TEST SUITE")
    logger.info("="*70)
    #
    # # Test 1: Causal Pathways
    # logger.info("\n\n" + "="*70)
    # logger.info("SUITE 1: CAUSAL PATHWAYS")
    # logger.info("="*70)
    # for tc in CAUSAL_PATHWAY_TESTS:
    #     try:
    #         test_causal_pathway(tc)
    #     except Exception as e:
    #         logger.error(f"❌ FAIL: {e}", exc_info=True)
    #
    # # Test 2: Interventions
    # logger.info("\n\n" + "="*70)
    # logger.info("SUITE 2: INTERVENTIONS (DiD, RD, Matching)")
    # logger.info("="*70)
    # for tc in INTERVENTION_TESTS:
    #     try:
    #         test_intervention(tc)
    #     except Exception as e:
    #         logger.error(f"❌ FAIL: {e}", exc_info=True)
    #
    # # Test 3: Discontinuity
    # logger.info("\n\n" + "="*70)
    # logger.info("SUITE 3: REGRESSION DISCONTINUITY")
    # logger.info("="*70)
    # for tc in DISCONTINUITY_TESTS:
    #     try:
    #         test_causal_pathway(tc)
    #     except Exception as e:
    #         logger.error(f"❌ FAIL: {e}", exc_info=True)
    #
    # # Test 4: Counterfactuals
    # logger.info("\n\n" + "="*70)
    # logger.info("SUITE 4: COUNTERFACTUAL SCENARIOS")
    # logger.info("="*70)
    # for tc in COUNTERFACTUAL_TESTS:
    #     try:
    #         test_counterfactual(tc)
    #     except Exception as e:
    #         logger.error(f"❌ FAIL: {e}", exc_info=True)
    #
    # # Test 5: Identification Challenges
    # logger.info("\n\n" + "="*70)
    # logger.info("SUITE 5: IDENTIFICATION CHALLENGES")
    # logger.info("="*70)
    # for tc in IDENTIFICATION_CHALLENGE_TESTS:
    #     try:
    #         test_identification_challenge(tc)
    #     except Exception as e:
    #         logger.error(f"❌ FAIL: {e}", exc_info=True)

    # Test 6: Complex Interventions
    logger.info("\n\n" + "="*70)
    logger.info("SUITE 6: COMPLEX MULTI-INTERVENTION SCENARIOS")
    logger.info("="*70)
    for tc in COMPLEX_INTERVENTION_TESTS:
        try:
            test_causal_pathway(tc)
        except Exception as e:
            logger.error(f"❌ FAIL: {e}", exc_info=True)
        break

    # # Test 7: Granularity
    # logger.info("\n\n" + "="*70)
    # logger.info("SUITE 7: DATA GRANULARITY & TEMPORAL ALIGNMENT")
    # logger.info("="*70)
    # for tc in GRANULARITY_TESTS:
    #     try:
    #         test_causal_pathway(tc)
    #     except Exception as e:
    #         logger.error(f"❌ FAIL: {e}", exc_info=True)

    logger.info("\n\n" + "="*70)
    logger.info("TEST SUITE COMPLETE")
    logger.info("="*70)
