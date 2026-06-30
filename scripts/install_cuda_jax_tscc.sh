#!/usr/bin/env bash
set -euo pipefail
# Run inside the conda environment used for STITCHCONT on a CUDA-capable TSCC GPU node.
# The package itself cannot install CUDA JAX on a remote cluster; this helper makes the environment reproducible.
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda12]"
python - <<'PY'
import json
import jax
report = {
    "jax_version": jax.__version__,
    "devices": [str(d) for d in jax.devices()],
    "platforms": [getattr(d, "platform", "unknown") for d in jax.devices()],
    "has_accelerator": any(getattr(d, "platform", "") in {"gpu", "tpu"} for d in jax.devices()),
}
print(json.dumps(report, indent=2))
if not report["has_accelerator"]:
    raise SystemExit("CUDA-enabled jaxlib did not expose a GPU/TPU device.")
PY
