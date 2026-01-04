"""
Vitals for Python: LLM Evaluation Framework
"""

from .task import Task, Sample, TaskMetrics, generate
from .scorers import (
    detect_includes,
    detect_match,
    detect_pattern,
    detect_exact,
    model_graded_qa,
    model_graded_fact
)

__all__ = [
    'Task',
    'Sample',
    'TaskMetrics',
    'generate',
    'detect_includes',
    'detect_match',
    'detect_pattern',
    'detect_exact',
    'model_graded_qa',
    'model_graded_fact'
]

__version__ = '0.1.0'
