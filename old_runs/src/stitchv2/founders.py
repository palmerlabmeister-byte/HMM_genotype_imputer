from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysam


@dataclass(slots=True)
class FounderPanel:
    chromosome: str
    positions: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    alt_prob: np.ndarray
    immutable_mask: np.ndarray

    @property
    def n_positions(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_founders(self) -> int:
        return int(self.alt_prob.shape[0])

    def with_extra_mutable_founders(
        self,
        target_n_founders: int,
        *,
        initial_alt_prob: float = 0.5,
    ) -> "FounderPanel":
        """Return a panel padded to target_n_founders with mutable founders.

        Loaded VCF/PLINK founders keep their current immutable/mutable status.
        Extra founders are initialized uniformly and marked mutable so the HMM
        can learn them during EM. The HMM later seeds collapsed mutable founders
        from observed site allele fractions when available.
        """
        target = int(target_n_founders)
        if target <= self.n_founders:
            return self
        n_extra = target - self.n_founders
        extra = np.full(
            (n_extra, self.n_positions),
            float(initial_alt_prob),
            dtype=np.float32,
        )
        return FounderPanel(
            chromosome=self.chromosome,
            positions=self.positions,
            ref=self.ref,
            alt=self.alt,
            alt_prob=np.concatenate([self.alt_prob.astype(np.float32, copy=False), extra], axis=0),
            immutable_mask=np.concatenate(
                [
                    self.immutable_mask.astype(bool, copy=False),
                    np.zeros(n_extra, dtype=bool),
                ],
                axis=0,
            ),
        )

    def slice(self, start: int, stop: int) -> "FounderPanel":
        return FounderPanel(
            chromosome=self.chromosome,
            positions=self.positions[start:stop],
            ref=self.ref[start:stop],
            alt=self.alt[start:stop],
            alt_prob=self.alt_prob[:, start:stop],
            immutable_mask=self.immutable_mask,
        )

    def to_parquet(self, path: str | Path, compression: str = "zstd") -> None:
        founder_ids = np.repeat(np.arange(self.n_founders), self.n_positions)
        positions = np.tile(self.positions, self.n_founders)
        alt_prob = self.alt_prob.reshape(-1)
        immutable = np.repeat(self.immutable_mask.astype(np.int8), self.n_positions)
        table = pa.table(
            {
                "chromosome": np.repeat(self.chromosome, founder_ids.shape[0]),
                "position": positions,
                "founder": founder_ids,
                "alt_prob": alt_prob,
                "immutable": immutable,
            }
        )
        pq.write_table(table, path, compression=compression)

    @classmethod
    def from_vcf(
        cls,
        vcf_path: str | Path,
        chromosome: str,
        positions_df: pd.DataFrame,
        immutable: bool = True,
    ) -> "FounderPanel":
        positions = positions_df["POS"].to_numpy(dtype=np.int64)
        ref = positions_df["REF"].fillna("N").astype(str).to_numpy()
        alt = positions_df["ALT"].fillna("N").astype(str).to_numpy()
        variant_file = pysam.VariantFile(str(vcf_path))
        samples = list(variant_file.header.samples)
        alt_prob = np.full((len(samples), len(positions)), 0.5, dtype=np.float32)
        pos_to_index = {int(pos): idx for idx, pos in enumerate(positions)}
        for record in variant_file.fetch(chromosome):
            idx = pos_to_index.get(int(record.pos))
            if idx is None:
                continue
            for sample_idx, sample_name in enumerate(samples):
                sample = record.samples[sample_name]
                gt = sample.get("GT")
                ds = sample.get("DS")
                if ds is not None:
                    alt_prob[sample_idx, idx] = np.clip(float(ds) / 2.0, 0.0, 1.0)
                elif gt is not None and None not in gt:
                    alt_prob[sample_idx, idx] = np.mean(gt)
        return cls(
            chromosome=chromosome,
            positions=positions,
            ref=ref,
            alt=alt,
            alt_prob=alt_prob,
            immutable_mask=np.full(len(samples), immutable, dtype=bool),
        )

    @classmethod
    def from_plink(
        cls,
        plink_prefix: str | Path,
        chromosome: str,
        positions_df: pd.DataFrame,
        immutable: bool = True,
    ) -> "FounderPanel":
        try:
            from .microarray import _normalize_chrom, _read_plink_bim, load_microarray_hardcalls_from_plink
        except ImportError as exc:
            raise ImportError("STITCHV2's internal PLINK loader is required for PLINK founder input.") from exc

        plink_prefix = Path(plink_prefix)
        hardcalls = load_microarray_hardcalls_from_plink(
            plink_prefix,
            chromosome=chromosome,
            positions_df=positions_df,
        )
        bim = _read_plink_bim(plink_prefix)
        bim = bim.loc[bim["chrom_norm"] == _normalize_chrom(chromosome)].copy()
        positions = positions_df["POS"].to_numpy(dtype=np.int64)
        ref = positions_df["REF"].fillna("N").astype(str).to_numpy()
        alt = positions_df["ALT"].fillna("N").astype(str).to_numpy()
        genetic_cm = _genetic_cm_from_positions_df(positions_df)
        if genetic_cm is None:
            genetic_cm = _interpolate_cm_from_bim(bim, positions)

        n_founders = int(hardcalls.dosage.shape[0])
        alt_prob = np.full((n_founders, len(positions)), 0.5, dtype=np.float32)
        values = np.asarray(hardcalls.dosage, dtype=np.float32) / 2.0
        finite = np.isfinite(values)
        alt_prob[finite] = np.clip(values[finite], 0.0, 1.0)
        return cls(
            chromosome=chromosome,
            positions=positions,
            ref=ref,
            alt=alt,
            alt_prob=alt_prob.astype(np.float32),
            immutable_mask=np.full(n_founders, immutable, dtype=bool),
            genetic_cm=genetic_cm,
        )


def load_founders(
    source_format: str,
    source_path: str | Path,
    chromosome: str,
    positions_df: pd.DataFrame,
    immutable: bool = True,
) -> FounderPanel:
    if source_format == "vcf":
        return FounderPanel.from_vcf(source_path, chromosome, positions_df, immutable=immutable)
    if source_format == "plink":
        return FounderPanel.from_plink(source_path, chromosome, positions_df, immutable=immutable)
    raise NotImplementedError(
        "Founder BAM support should be implemented from a likelihood-based pileup, "
        "not a hard genotype loader. Use VCF/PLINK for now or extend FounderPanel."
    )
