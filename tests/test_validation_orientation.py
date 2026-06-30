from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_allele_orientation_audit_flips_truth_when_run_alt_is_plink_a1(tmp_path: Path):
    pytest.importorskip("pyarrow")
    from stitchcont.validation import _orientation_audit_and_flip

    run = tmp_path / "run"
    out = tmp_path / "validation"
    run.mkdir()
    out.mkdir()
    pd.DataFrame({"CHR": ["chr1", "chr1"], "POS": [100, 200], "REF": ["A", "C"], "ALT": ["G", "T"]}).to_parquet(run / "positions.parquet", index=False)
    prefix = tmp_path / "truth"
    # PLINK hardcall loader counts A2. At POS=100, A2=ALT -> no flip. At POS=200, A2=REF -> truth must flip.
    (prefix.with_suffix(".bim")).write_text("1 rs1 0 100 A G\n1 rs2 0 200 T C\n", encoding="utf-8")
    truth = np.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    audited, summary = _orientation_audit_and_flip(
        run_output_dir=run,
        truth_plink=prefix,
        chromosome="chr1",
        positions=np.asarray([100, 200], dtype=np.int64),
        truth_dosage=truth,
        fail_on_mismatch=False,
        output_dir=out,
    )
    np.testing.assert_allclose(audited[:, 0], truth[:, 0])
    np.testing.assert_allclose(audited[:, 1], 2.0 - truth[:, 1])
    assert summary["n_flipped_variants"] == 1
