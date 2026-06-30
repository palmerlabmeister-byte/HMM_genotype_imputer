"""stitchcont: streaming STITCH-style low-coverage imputation."""
from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _package_version

try:
    __version__ = _package_version("stitchcont")
except PackageNotFoundError:  # pragma: no cover - source tree before install.
    __version__ = "0.1.0"

_LAZY_EXPORTS = {
    "FounderConfig": (".config", "FounderConfig"),
    "HMMConfig": (".config", "HMMConfig"),
    "IOConfig": (".config", "IOConfig"),
    "PipelineConfig": (".config", "PipelineConfig"),
    "FounderPanel": (".founders", "FounderPanel"),
    "HMMArtifacts": (".hmm", "HMMArtifacts"),
    "JAXStitchHMM": (".hmm", "JAXStitchHMM"),
    "StitchPipeline": (".pipeline", "StitchPipeline"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError as exc:  # pragma: no cover
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


__all__ = ["__version__", *_LAZY_EXPORTS.keys()]
