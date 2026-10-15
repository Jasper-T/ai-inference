"""Test utilities that execute the same ordered pipelines as Model."""

from types import SimpleNamespace

from ai_inference import Pipeline
from ai_inference.processing.pipeline import validate_connection


def options(data, name):
    """Mutable parameters of one explicit operation in a config or dump."""
    sections = (
        (data.preprocess, data.postprocess)
        if hasattr(data, "preprocess")
        else (data["preprocess"], data["postprocess"])
    )
    matches = [entry[name] for section in sections for entry in section["steps"] if name in entry]
    if len(matches) != 1:
        raise ValueError(f"expected one {name} operation")
    return matches[0]


def pipelines(config):
    pre = Pipeline("preprocess", config.preprocess, config)
    post = Pipeline("postprocess", config.postprocess, config)
    validate_connection(pre, post)
    return SimpleNamespace(config=config, preprocess=pre, postprocess=post)
