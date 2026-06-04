"""Round-trip test for PLINK founder loading via the bundled stitchv2.npplink
reader (replaces the previously required external ``npplink`` package).

Writes a tiny .bed/.bim/.fam trio, loads it through FounderPanel.from_plink, and
checks the founder alt-probabilities equal genotype/2 (missing -> 0.5 prior).

Requires dask (a core dependency); skipped if unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dask")

from stitchv2.founders import FounderPanel
from stitchv2.npplink import _pack_plink_variant


def _write_plink(prefix, genotypes: np.ndarray, positions, chrom="1", a0="A", a1="C"):
    n_samples, n_variants = genotypes.shape
    fam = pd.DataFrame(
        {
            "fid": [f"S{i}" for i in range(n_samples)],
            "iid": [f"S{i}" for i in range(n_samples)],
            "father": "0",
            "mother": "0",
            "gender": 0,
            "trait": -9,
        }
    )
    bim = pd.DataFrame(
        {
            "chrom": chrom,
            "snp": [f"{chrom}:{p}" for p in positions],
            "cm": 0.0,
            "pos": positions,
            "a0": a0,
            "a1": a1,
        }
    )
    fam.to_csv(f"{prefix}.fam", sep="\t", header=False, index=False)
    bim.to_csv(f"{prefix}.bim", sep="\t", header=False, index=False)
    with open(f"{prefix}.bed", "wb") as fh:
        fh.write(bytes([0x6C, 0x1B, 0x01]))
        for v in range(n_variants):
            fh.write(_pack_plink_variant(genotypes[:, v]))


def test_from_plink_roundtrip(tmp_path):
    nan = np.nan
    geno = np.array(
        [
            [0, 1, 2, nan],
            [2, 2, 0, 1],
            [1, nan, 1, 0],
        ],
        dtype=np.float32,
    )
    positions = [100, 200, 300, 400]
    prefix = str(tmp_path / "founders")
    _write_plink(prefix, geno, positions)

    positions_df = pd.DataFrame(
        {"POS": positions, "REF": ["A"] * 4, "ALT": ["C"] * 4}
    )
    panel = FounderPanel.from_plink(prefix, "1", positions_df, immutable=True)

    expected = np.where(np.isnan(geno), 0.5, geno / 2.0).astype(np.float32)
    assert panel.alt_prob.shape == (3, 4)
    np.testing.assert_allclose(panel.alt_prob, expected, atol=1e-6)
    assert panel.immutable_mask.all()


def test_chr_prefix_is_tolerated(tmp_path):
    geno = np.array([[0, 2], [2, 0]], dtype=np.float32)
    positions = [10, 20]
    prefix = str(tmp_path / "founders_chr")
    _write_plink(prefix, geno, positions, chrom="chr1")
    positions_df = pd.DataFrame({"POS": positions, "REF": ["A", "A"], "ALT": ["C", "C"]})
    # Request "1" while the BIM stores "chr1": loader strips the prefix on both sides.
    panel = FounderPanel.from_plink(prefix, "1", positions_df)
    np.testing.assert_allclose(panel.alt_prob, geno / 2.0, atol=1e-6)
