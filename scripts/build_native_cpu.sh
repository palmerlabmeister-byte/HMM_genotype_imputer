#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from stitchcont.native_cpu import build_native_library
print(build_native_library(force=True))
PY
