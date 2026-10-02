"""
SEC Intelligence Platform - Package initialization
"""

__version__ = "0.1.0"
__author__ = "SEC Intelligence Team"

# Public names are resolved lazily (PEP 562) so that importing a light submodule,
# e.g. sec_intelligence.agent.planner, does not pull in the database, vector store
# and embedding stacks. `from sec_intelligence import SECIntelligencePlatform` still works.
_LAZY = {
    "SECIntelligencePlatform": "sec_intelligence.platform",
    "create_platform": "sec_intelligence.platform",
    "FilingType": "sec_intelligence.schemas",
    "QuestionAnswerRequest": "sec_intelligence.schemas",
    "QuestionAnswerResponse": "sec_intelligence.schemas",
    "ComparisonRequest": "sec_intelligence.schemas",
    "ComparisonResult": "sec_intelligence.schemas",
    "ExtractedIntelligence": "sec_intelligence.schemas",
    "ExtractedRisk": "sec_intelligence.schemas",
    "FinancialMetric": "sec_intelligence.schemas",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
