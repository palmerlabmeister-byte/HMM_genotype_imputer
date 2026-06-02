from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysam


def _usable_genetic_cm(values: np.ndarray | pd.Series | None) -> bool:
    if values is None:
        return False
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return False
    return bool(np.nanmax(finite) - np.nanmin(finite) > 0.0 and np.nanmax(np.abs(finite)) > 0.0)


def _genetic_cm_from_positions_df(positions_df: pd.DataFrame) -> np.ndarray | None:
    if "CM" not in positions_df.columns:
        return None
    cm = pd.to_numeric(positions_df["CM"], errors="coerce").to_numpy(dtype=np.float32)
    if not _usable_genetic_cm(cm):
        return None
    return cm.astype(np.float32, copy=False)


def _find_bim_cm_column(bim: pd.DataFrame) -> str | None:
    candidates = {"cm", "centimorgan", "genetic_cm", "genetic_distance"}
    for column in bim.columns:
        if str(column).strip().lower() in candidates:
            return str(column)
    return None


def _interpolate_cm_from_bim(bim: pd.DataFrame, positions: np.ndarray) -> np.ndarray | None:
    cm_col = _find_bim_cm_column(bim)
    if cm_col is None or "pos" not in bim.columns:
        return None
    bim_pos = pd.to_numeric(bim["pos"], errors="coerce").to_numpy(dtype=np.float64)
    bim_cm = pd.to_numeric(bim[cm_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(bim_pos) & np.isfinite(bim_cm)
    if int(np.sum(valid)) < 2:
        return None
    known_pos = bim_pos[valid]
    known_cm = bim_cm[valid]
    order = np.argsort(known_pos)
    known_pos = known_pos[order]
    known_cm = known_cm[order]
    unique_pos, unique_idx = np.unique(known_pos, return_index=True)
    known_cm = known_cm[unique_idx]
    if not _usable_genetic_cm(known_cm):
        return None
    cm = np.interp(positions.astype(np.float64), unique_pos, known_cm).astype(np.float32)
    return cm if _usable_genetic_cm(cm) else None


@dataclass(slots=True)
class FounderPanel:
    chromosome: str
    positions: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    alt_prob: np.ndarray
    immutable_mask: np.ndarray
    genetic_cm: np.ndarray | None = None

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
            genetic_cm=None if self.genetic_cm is None else self.genetic_cm.astype(np.float32, copy=True),
        )

    def slice(self, start: int, stop: int) -> "FounderPanel":
        return FounderPanel(
            chromosome=self.chromosome,
            positions=self.positions[start:stop],
            ref=self.ref[start:stop],
            alt=self.alt[start:stop],
            alt_prob=self.alt_prob[:, start:stop],
            immutable_mask=self.immutable_mask,
            genetic_cm=None if self.genetic_cm is None else self.genetic_cm[start:stop],
        )

    def to_parquet(self, path: str | Path, compression: str = "zstd") -> None:
        founder_ids = np.repeat(np.arange(self.n_founders), self.n_positions)
        positions = np.tile(self.positions, self.n_founders)
        alt_prob = self.alt_prob.reshape(-1)
        immutable = np.repeat(self.immutable_mask.astype(np.int8), self.n_positions)
        columns = {
            "chromosome": np.repeat(self.chromosome, founder_ids.shape[0]),
            "position": positions,
            "founder": founder_ids,
            "alt_prob": alt_prob,
            "immutable": immutable,
        }
        if self.genetic_cm is not None and _usable_genetic_cm(self.genetic_cm):
            columns["genetic_cm"] = np.tile(self.genetic_cm.astype(np.float32, copy=False), self.n_founders)
        table = pa.table(columns)
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
        genetic_cm = _genetic_cm_from_positions_df(positions_df)
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
            genetic_cm=genetic_cm,
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
            from npplink import Plink
        except ImportError as exc:
            raise ImportError("npplink is required for PLINK founder input.") from exc

        plink = Plink(str(plink_prefix))
        bim = plink.get_bim()
        chr_mask = bim["chrom"].astype(str) == str(chromosome).replace("chr", "")
        bim = bim.loc[chr_mask].copy()
        bim_variant_index = bim.index.to_numpy(dtype=np.int64, copy=True)
        bim = bim.reset_index(drop=True)
        geno = plink.get_geno()
        positions = positions_df["POS"].to_numpy(dtype=np.int64)
        ref = positions_df["REF"].fillna("N").astype(str).to_numpy()
        alt = positions_df["ALT"].fillna("N").astype(str).to_numpy()
        genetic_cm = _genetic_cm_from_positions_df(positions_df)
        if genetic_cm is None:
            genetic_cm = _interpolate_cm_from_bim(bim, positions)
        pos_to_input = {int(pos): idx for idx, pos in enumerate(positions)}
        alt_prob = np.full((geno.shape[0], len(positions)), 0.5, dtype=np.float32)
        for bim_idx, pos in zip(bim_variant_index.tolist(), bim["pos"].to_numpy(dtype=np.int64), strict=False):
            target_idx = pos_to_input.get(int(pos))
            if target_idx is None:
                continue
            alt_prob[:, target_idx] = geno[:, int(bim_idx)] / 2.0
        return cls(
            chromosome=chromosome,
            positions=positions,
            ref=ref,
            alt=alt,
            alt_prob=alt_prob.astype(np.float32),
            immutable_mask=np.full(geno.shape[0], immutable, dtype=bool),
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
