"""Reproducible, label-free Lecture 2 feature extraction and benchmarks.

See ``lectures/lec2/feature_benchmark/README.md`` for the protocol and examples.
"""

__all__ = ["ExtractionResult", "extract_features", "run_benchmark", "list_methods"]


def __getattr__(name):
    """Load lazily so the CLI can configure threads before importing numerical libs."""
    if name == "list_methods":
        from sktime.benchmarking.feature_extraction._methods import list_methods

        return list_methods
    if name in __all__:
        from sktime.benchmarking.feature_extraction import _engine

        return getattr(_engine, name)
    raise AttributeError(name)
