"""Native extension modules for stitchcont.

The preferred native backend is compiled at install time via pybind11.  The
ctypes/manual-build path in :mod:`stitchcont.native_cpu` is retained as a
fallback for development environments that did not build the extension.
"""
