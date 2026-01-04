"""
StatLingua for Python: LLM-powered statistical output explanation
"""

from .explainer import (
    StatLinguaExplainer,
    StatLinguaExplanation,
    explain,
    AudienceType,
    VerbosityType,
    StyleType
)

__all__ = [
    'StatLinguaExplainer',
    'StatLinguaExplanation',
    'explain',
    'AudienceType',
    'VerbosityType',
    'StyleType'
]

__version__ = '0.1.0'
