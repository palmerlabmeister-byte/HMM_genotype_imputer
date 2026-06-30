from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def _import_pybind11_include() -> str:
    import pybind11

    return pybind11.get_include()


def _openmp_flags() -> tuple[list[str], list[str]]:
    system = platform.system()
    if system == "Linux":
        return ["-fopenmp"], ["-fopenmp"]
    if system == "Windows":
        return ["/openmp"], []
    # macOS needs libomp from Homebrew/conda and custom include/library paths.
    # Keep it off by default so editable installs do not fail unexpectedly.
    return [], []


def _compile_args() -> tuple[list[str], list[str]]:
    compile_omp, link_omp = _openmp_flags()
    system = platform.system()
    if system == "Windows":
        return ["/O2", "/std:c++17", *compile_omp], [*link_omp]
    args = ["-O3", "-std=c++17", "-fPIC"]
    if os.environ.get("STITCHCONT_NATIVE_MARCH_NATIVE", "1") not in {"0", "false", "False"}:
        args.append("-march=native")
    if os.environ.get("STITCHCONT_NATIVE_FAST_MATH", "0") not in {"0", "false", "False"}:
        args.append("-ffast-math")
    args.extend(compile_omp)
    return args, [*link_omp]


compile_args, link_args = _compile_args()

extensions = [
    Extension(
        "stitchcont.native._hs_k8_unordered",
        sources=[
            "src/stitchcont/native/pybind_hs_k8_unordered.cpp",
            "src/stitchcont/native/hs_k8_unordered.cpp",
        ],
        include_dirs=[
            _import_pybind11_include(),
            np.get_include(),
            "src/stitchcont/native",
        ],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]


class OptionalNativeBuildExt(build_ext):
    """Build native extensions, with an opt-out for pure-Python installs.

    By default the pybind11 K8 backend is compiled during `pip install -e .`.
    Set STITCHCONT_SKIP_NATIVE=1 to skip this step, for example on login nodes
    without a compiler.  The runtime still has Numba/NumPy fallbacks.

    On Linux, the first build attempt uses OpenMP.  If that fails, the command
    retries the same extension without OpenMP so editable installs still work on
    systems whose default compiler does not provide `-fopenmp`.
    """

    def run(self):
        if os.environ.get("STITCHCONT_SKIP_NATIVE", "0") in {"1", "true", "True"}:
            self.extensions = []
            return
        super().run()

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception:
            if "-fopenmp" not in ext.extra_compile_args and "-fopenmp" not in ext.extra_link_args:
                raise
            ext.extra_compile_args = [arg for arg in ext.extra_compile_args if arg != "-fopenmp"]
            ext.extra_link_args = [arg for arg in ext.extra_link_args if arg != "-fopenmp"]
            super().build_extension(ext)


setup(
    ext_modules=extensions,
    cmdclass={"build_ext": OptionalNativeBuildExt},
)
