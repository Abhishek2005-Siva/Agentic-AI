"""
SEC Intelligence Platform - Package initialization
"""

__version__ = "0.1.0"
__author__ = "SEC Intelligence Team"

from sec_intelligence.platform import SECIntelligencePlatform, create_platform
from sec_intelligence.schemas import (
    FilingType,
    QuestionAnswerRequest,
    QuestionAnswerResponse,
    ComparisonRequest,
    ComparisonResult,
    ExtractedIntelligence,
    ExtractedRisk,
    FinancialMetric,
)

__all__ = [
    "SECIntelligencePlatform",
    "create_platform",
    "FilingType",
    "QuestionAnswerRequest",
    "QuestionAnswerResponse",
    "ComparisonRequest",
    "ComparisonResult",
    "ExtractedIntelligence",
    "ExtractedRisk",
    "FinancialMetric",
]
