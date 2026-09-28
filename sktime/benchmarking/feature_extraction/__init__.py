"""Reproducible, label-free Lecture 2 feature extraction and benchmarks.

See ``lectures/lec2/feature_benchmark/README.md`` for the protocol and examples.
"""

from sktime.benchmarking.feature_extraction._engine import (
    ExtractionResult,
    extract_features,
    run_benchmark,
)
from sktime.benchmarking.feature_extraction._methods import list_methods

__all__ = ["ExtractionResult", "extract_features", "run_benchmark", "list_methods"]
