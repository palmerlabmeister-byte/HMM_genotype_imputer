from __future__ import annotations

import json
import re
from pathlib import Path


def _rewrite_name(name: str) -> str:
    base = Path(name).name.strip()
    if not base:
        return "stitchcont_output"
    normalized = re.sub(r"\s+", "_", base)
    if re.search(r"(?i)\bstitch\b", normalized):
        normalized = re.sub(r"(?i)\bstitch\b", "stitchcont", normalized)
    elif not normalized.lower().startswith("stitchcont"):
        normalized = f"stitchcont_{normalized}"
    return normalized


def reformat_stitch_filenames(
    filenames: list[str],
    *,
    output_path: str | Path | None = None,
) -> list[str]:
    out = [_rewrite_name(str(x)) for x in filenames]
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"original": str(src), "stitchcont": dst} for src, dst in zip(filenames, out, strict=False)]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
