"""News vs SEC result comparison components."""

from orchestrator.comparison.discrepancy import (
    Agreement,
    ComparisonResult,
    Discrepancy,
    DiscrepancyDetector,
    DiscrepancySeverity,
    DiscrepancyType,
)
from orchestrator.comparison.signals import (
    ExtractedSignal,
    SignalDirection,
    SignalExtractionResult,
    SignalExtractor,
    SignalType,
)

__all__ = [
    "Agreement",
    "ComparisonResult",
    "Discrepancy",
    "DiscrepancyDetector",
    "DiscrepancySeverity",
    "DiscrepancyType",
    "ExtractedSignal",
    "SignalDirection",
    "SignalExtractionResult",
    "SignalExtractor",
    "SignalType",
]
