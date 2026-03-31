#!/usr/bin/env python
"""
Minimal test to verify FeasibilityReport attribute fix.
NO LLM CALLS — just tests the data structure.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from reasoning.metadata_analyzer import FeasibilityReport, DatasetInfo

def test_feasibility_report_attributes():
    """Test that FeasibilityReport has correct attributes."""
    logger.info("=" * 60)
    logger.info("TEST: FeasibilityReport Attributes")
    logger.info("=" * 60)

    # Create a minimal report
    dataset = DatasetInfo(
        title="Test Dataset",
        description="Test",
        sector="health",
        granularity_levels=["state"],
        tags=["health"],
        frequency="annual",
        total_count=1000
    )

    report = FeasibilityReport(
        found_datasets=[dataset],
        coverage_score=0.75,
        gaps=["Missing district-level data"],
        granularity_matched=True,
        recommendations=["Aggregate to state level"]
    )

    # Verify attributes exist and work
    logger.info(f"\n✓ coverage_score: {report.coverage_score}")
    assert hasattr(report, 'coverage_score'), "Missing coverage_score"
    assert report.coverage_score == 0.75, "Wrong value"

    logger.info(f"✓ found_datasets: {len(report.found_datasets)}")
    assert len(report.found_datasets) == 1, "Wrong count"

    logger.info(f"✓ gaps: {report.gaps}")
    assert len(report.gaps) == 1, "Wrong gaps"

    logger.info(f"✓ granularity_matched: {report.granularity_matched}")
    assert report.granularity_matched is True, "Wrong value"

    logger.info(f"✓ recommendations: {report.recommendations}")
    assert len(report.recommendations) == 1, "Wrong recommendations"

    # Verify OLD attribute doesn't exist (this was the bug)
    assert not hasattr(report, 'feasibility'), "Old 'feasibility' attribute still exists!"
    logger.info(f"✓ Old 'feasibility' attribute correctly removed")

    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL CHECKS PASSED")
    logger.info("=" * 60)

if __name__ == "__main__":
    test_feasibility_report_attributes()
