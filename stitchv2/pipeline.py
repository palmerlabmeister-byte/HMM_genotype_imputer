from __future__ import annotations

import gc
import json
import os
import resource
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from .calibration import (
    calibrate_genotype_posterior,
    calibrate_genotype_posterior_block_context,
    calibrate_genotype_posterior_full_stack,
    dosage_to_genotype_posterior,
    genotype_call_from_posterior,
)
from .config import PipelineConfig
from .dask_executor import (
    DaskHMMTaskResult,
    plan_dask_chunks,
    run_hmm_leaf_task,
    write_task_stream_artifact,
)
from .founders import FounderPanel, load_founders
from .hmm import HMMArtifacts, JAXStitchHMM
from .io import iter_position_blocks, load_positions, validate_samples, write_parquet
from .microarray import (
    align_microarray_to_samples,
    load_microarray_hardcalls_from_plink,
    load_microarray_hardcalls_from_sample_plink_paths,
)
from .pedigree import smooth_dosage_with_pedigree
from .pileup import PysamReadExtractor, ReadEvidenceBlock


def _current_rss_mb() -> float:
    statm_path = Path("/proc/self/statm")
    if statm_path.exists():
        try:
            fields = statm_path.read_text(encoding="utf-8").split()
            if len(fields) >= 2:
                page_size = os.sysconf("SC_PAGE_SIZE")
                return (int(fields[1]) * float(page_size)) / (1024.0 * 1024.0)
        except Exception:
            pass
    # Fallback to peak RSS when current RSS is unavailable.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class StitchPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.io_config = config.io()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._microarray_dosage: np.ndarray | None = None
        self._sample_ploidy: np.ndarray | None = None
        self.hmm = JAXStitchHMM(config.hmm())
        self.read_extractor = PysamReadExtractor(
            config.chromosome,
            mode=config.read_mode,
            read_stream_backend=config.read_stream_backend,
            merge_fragments_by_query=config.merge_fragments_by_query,
            read_batch_size=config.read_batch_size,
            io_workers=config.io_workers,
            htslib_threads_per_file=config.htslib_threads_per_file,
            memory_map_read_matrices=config.memory_map_read_matrices,
            memory_map_dir=config.memory_map_dir,
            subsample_seed=int(config.random_seed),
        )

    def prepare_inputs(
        self,
        samples: pd.DataFrame,
        pedigree=None,
        founder_panel: FounderPanel | None = None,
    ) -> FounderPanel:
        samples = validate_samples(samples)
        positions_df = load_positions(
            self.config.positions_path,
            self.config.chromosome,
            start=self.config.chromosome_start,
            end=self.config.chromosome_end,
        )
        if positions_df.empty:
            raise ValueError(
                f"No positions available for chromosome={self.config.chromosome} "
                f"with start={self.config.chromosome_start} end={self.config.chromosome_end}."
            )
        self._microarray_dosage = None
        if self.config.microarray_plink_path:
            hardcalls = load_microarray_hardcalls_from_plink(
                self.config.microarray_plink_path,
                chromosome=self.config.chromosome,
                positions_df=positions_df,
            )
            samples, self._microarray_dosage = align_microarray_to_samples(
                samples,
                hardcalls,
                add_missing_samples=bool(self.config.microarray_add_samples),
                generation_default=self.config.microarray_generation_default,
            )
        if "plink_path" in samples.columns:
            mapped = load_microarray_hardcalls_from_sample_plink_paths(
                samples,
                chromosome=self.config.chromosome,
                positions_df=positions_df,
                plink_path_column="plink_path",
            )
            if self._microarray_dosage is None:
                self._microarray_dosage = mapped.dosage
            else:
                sample_micro = mapped.dosage
                valid = np.isfinite(sample_micro)
                if np.any(valid):
                    self._microarray_dosage[valid] = sample_micro[valid]
        self._sample_ploidy = self._resolve_sample_ploidy(samples)
        samples = samples.copy()
        samples["ploidy"] = self._sample_ploidy.astype(np.int16, copy=False)
        write_parquet(
            pa.Table.from_pandas(samples),
            self.output_dir / "samples.parquet",
            self.config.compression,
            self.config.compression_level,
        )
        write_parquet(
            pa.Table.from_pandas(positions_df),
            self.output_dir / "positions.parquet",
            self.config.compression,
            self.config.compression_level,
        )

        if founder_panel is None:
            if self.config.founder.source_path is None:
                raise ValueError("Founder source_path is required unless founder_panel is provided.")
            founder_panel = load_founders(
                source_format=self.config.founder.source_format,
                source_path=self.config.founder.source_path,
                chromosome=self.config.chromosome,
                positions_df=positions_df,
                immutable=self.config.founder.immutable,
            )
        founder_panel.to_parquet(self.output_dir / "founders.parquet", compression=self.config.compression)
        self._run_blocks(samples, positions_df, founder_panel, pedigree)
        return founder_panel

    def _resolve_sample_ploidy(self, samples: pd.DataFrame) -> np.ndarray:
        default_ploidy = int(self.config.ploidy)
        if default_ploidy < 0:
            raise ValueError(f"ploidy must be >= 0, got {default_ploidy}")
        out = np.full(len(samples), default_ploidy, dtype=np.int16)
        male_ploidy = self.config.ploidy_males
        female_ploidy = self.config.ploidy_females
        if male_ploidy is None and female_ploidy is None:
            return out
        if male_ploidy is None or female_ploidy is None:
            raise ValueError("Both ploidy_males and ploidy_females must be provided together.")
        male_ploidy = int(male_ploidy)
        female_ploidy = int(female_ploidy)
        if male_ploidy < 0 or female_ploidy < 0:
            raise ValueError("ploidy_males and ploidy_females must be >= 0.")
        if "sex" not in samples.columns:
            raise ValueError("samples table must contain a 'sex' column when ploidy_males/ploidy_females are used.")
        sex = samples["sex"].astype(str).str.strip().str.lower()
        male_mask = sex.isin({"m", "male", "1", "xy"}).to_numpy(dtype=bool)
        female_mask = sex.isin({"f", "female", "2", "xx"}).to_numpy(dtype=bool)
        out[male_mask] = np.int16(male_ploidy)
        out[female_mask] = np.int16(female_ploidy)
        return out

    @staticmethod
    def _mask_genotype_posterior_by_ploidy(
        posterior: np.ndarray,
        sample_ploidy: np.ndarray,
    ) -> np.ndarray:
        gp = posterior.astype(np.float32, copy=True)
        for sample_idx, ploidy in enumerate(sample_ploidy.astype(np.int16, copy=False).tolist()):
            if int(ploidy) <= 0:
                gp[sample_idx] = np.nan
                continue
            n_valid = min(int(ploidy) + 1, gp.shape[2])
            if n_valid < gp.shape[2]:
                gp[sample_idx, :, n_valid:] = 0.0
        row_sum = np.nansum(gp, axis=2, keepdims=True)
        valid = np.isfinite(row_sum) & (row_sum > 0.0)
        gp = np.divide(gp, np.clip(row_sum, 1e-12, None), out=np.full_like(gp, np.nan), where=valid)
        return gp.astype(np.float32, copy=False)

    @staticmethod
    def _build_full_sample_fragment_offsets(
        read_fragment_offsets: np.ndarray,
        has_bam_mask: np.ndarray,
    ) -> np.ndarray:
        n_samples = int(has_bam_mask.shape[0])
        full_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        read_idx = 0
        for sample_idx in range(n_samples):
            if bool(has_bam_mask[sample_idx]):
                n_frag = int(read_fragment_offsets[read_idx + 1] - read_fragment_offsets[read_idx])
                read_idx += 1
            else:
                n_frag = 0
            full_offsets[sample_idx + 1] = full_offsets[sample_idx] + n_frag
        return full_offsets

    @staticmethod
    def _inject_microarray_hard_calls(
        evidence: ReadEvidenceBlock,
        microarray_dosage_block: np.ndarray,
        *,
        hard_call_weight: int,
    ) -> None:
        if microarray_dosage_block.size == 0:
            return
        calls = np.rint(microarray_dosage_block).astype(np.float32, copy=False)
        valid = np.isfinite(calls)
        if not np.any(valid):
            return
        gt = np.clip(calls, 0.0, 2.0).astype(np.int8, copy=False)
        weight = max(int(hard_call_weight), 1)
        w_ref = weight // 2
        w_alt = weight - w_ref

        mask0 = valid & (gt == 0)
        mask1 = valid & (gt == 1)
        mask2 = valid & (gt == 2)

        if np.any(mask0):
            evidence.ref_count[mask0] = np.clip(
                evidence.ref_count[mask0].astype(np.int32) + weight,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            evidence.ref_weight[mask0] += float(weight)
            evidence.depth[mask0] = np.clip(
                evidence.depth[mask0].astype(np.int32) + weight,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
        if np.any(mask1):
            evidence.ref_count[mask1] = np.clip(
                evidence.ref_count[mask1].astype(np.int32) + w_ref,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            evidence.alt_count[mask1] = np.clip(
                evidence.alt_count[mask1].astype(np.int32) + w_alt,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            evidence.ref_weight[mask1] += float(w_ref)
            evidence.alt_weight[mask1] += float(w_alt)
            evidence.depth[mask1] = np.clip(
                evidence.depth[mask1].astype(np.int32) + weight,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
        if np.any(mask2):
            evidence.alt_count[mask2] = np.clip(
                evidence.alt_count[mask2].astype(np.int32) + weight,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            evidence.alt_weight[mask2] += float(weight)
            evidence.depth[mask2] = np.clip(
                evidence.depth[mask2].astype(np.int32) + weight,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)

    def _build_full_sample_evidence(
        self,
        *,
        block_id: int,
        chromosome: str,
        positions: np.ndarray,
        ref: np.ndarray,
        alt: np.ndarray,
        sample_ids: np.ndarray,
        has_bam_mask: np.ndarray,
        read_evidence: ReadEvidenceBlock | None,
    ) -> ReadEvidenceBlock:
        n_samples = int(sample_ids.shape[0])
        n_positions = int(positions.shape[0])
        shape = (n_samples, n_positions)
        ref_count = np.zeros(shape, dtype=np.uint16)
        alt_count = np.zeros(shape, dtype=np.uint16)
        other_count = np.zeros(shape, dtype=np.uint16)
        depth = np.zeros(shape, dtype=np.uint16)
        ref_weight = np.zeros(shape, dtype=np.float32)
        alt_weight = np.zeros(shape, dtype=np.float32)
        other_weight = np.zeros(shape, dtype=np.float32)
        n_overlapping_reads = np.zeros(n_samples, dtype=np.int32)

        if read_evidence is not None:
            bam_idx = np.flatnonzero(has_bam_mask)
            ref_count[bam_idx] = read_evidence.ref_count
            alt_count[bam_idx] = read_evidence.alt_count
            other_count[bam_idx] = read_evidence.other_count
            depth[bam_idx] = read_evidence.depth
            ref_weight[bam_idx] = read_evidence.ref_weight
            alt_weight[bam_idx] = read_evidence.alt_weight
            other_weight[bam_idx] = read_evidence.other_weight
            n_overlapping_reads[bam_idx] = read_evidence.n_overlapping_reads
            fragment_sample_offsets = self._build_full_sample_fragment_offsets(
                read_evidence.fragment_sample_offsets,
                has_bam_mask,
            )
            fragment_center_idx = read_evidence.fragment_center_idx
            fragment_obs_offsets = read_evidence.fragment_obs_offsets
            fragment_obs_pos_idx = read_evidence.fragment_obs_pos_idx
            fragment_obs_code = read_evidence.fragment_obs_code
            fragment_obs_qual = read_evidence.fragment_obs_qual
        else:
            fragment_sample_offsets = np.zeros(n_samples + 1, dtype=np.int64)
            fragment_center_idx = np.empty((0,), dtype=np.int32)
            fragment_obs_offsets = np.zeros(1, dtype=np.int64)
            fragment_obs_pos_idx = np.empty((0,), dtype=np.int32)
            fragment_obs_code = np.empty((0,), dtype=np.int8)
            fragment_obs_qual = np.empty((0,), dtype=np.uint8)

        return ReadEvidenceBlock(
            block_id=block_id,
            chromosome=chromosome,
            positions=positions,
            ref=ref,
            alt=alt,
            ref_count=ref_count,
            alt_count=alt_count,
            other_count=other_count,
            depth=depth,
            ref_weight=ref_weight,
            alt_weight=alt_weight,
            other_weight=other_weight,
            sample_ids=sample_ids,
            n_overlapping_reads=n_overlapping_reads,
            fragment_sample_offsets=fragment_sample_offsets,
            fragment_center_idx=fragment_center_idx,
            fragment_obs_offsets=fragment_obs_offsets,
            fragment_obs_pos_idx=fragment_obs_pos_idx,
            fragment_obs_code=fragment_obs_code,
            fragment_obs_qual=fragment_obs_qual,
            memmap_dir=None,
        )

    @staticmethod
    def _subset_fragments_by_sample_indices(
        sample_indices: np.ndarray,
        fragment_sample_offsets: np.ndarray,
        fragment_center_idx: np.ndarray,
        fragment_obs_offsets: np.ndarray,
        fragment_obs_pos_idx: np.ndarray,
        fragment_obs_code: np.ndarray,
        fragment_obs_qual: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        n_selected = int(sample_indices.size)
        local_sample_offsets = np.zeros(n_selected + 1, dtype=np.int64)
        if n_selected == 0:
            return (
                local_sample_offsets,
                np.empty((0,), dtype=np.int32),
                np.zeros(1, dtype=np.int64),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int8),
                np.empty((0,), dtype=np.uint8) if fragment_obs_qual is not None else None,
            )

        center_chunks: list[np.ndarray] = []
        pos_chunks: list[np.ndarray] = []
        code_chunks: list[np.ndarray] = []
        qual_chunks: list[np.ndarray] = []
        obs_cursor = 0
        obs_offsets = [0]
        for out_idx, sample_idx in enumerate(sample_indices.tolist()):
            frag_start = int(fragment_sample_offsets[int(sample_idx)])
            frag_stop = int(fragment_sample_offsets[int(sample_idx) + 1])
            local_sample_offsets[out_idx + 1] = local_sample_offsets[out_idx] + (frag_stop - frag_start)
            if frag_stop <= frag_start:
                continue
            center_chunks.append(fragment_center_idx[frag_start:frag_stop].astype(np.int32, copy=False))
            for frag in range(frag_start, frag_stop):
                o0 = int(fragment_obs_offsets[frag])
                o1 = int(fragment_obs_offsets[frag + 1])
                n_obs = o1 - o0
                if n_obs > 0:
                    pos_chunks.append(fragment_obs_pos_idx[o0:o1].astype(np.int32, copy=False))
                    code_chunks.append(fragment_obs_code[o0:o1].astype(np.int8, copy=False))
                    if fragment_obs_qual is not None:
                        qual_chunks.append(fragment_obs_qual[o0:o1].astype(np.uint8, copy=False))
                obs_cursor += n_obs
                obs_offsets.append(obs_cursor)

        centers = np.concatenate(center_chunks, axis=0) if center_chunks else np.empty((0,), dtype=np.int32)
        obs_pos = np.concatenate(pos_chunks, axis=0) if pos_chunks else np.empty((0,), dtype=np.int32)
        obs_code = np.concatenate(code_chunks, axis=0) if code_chunks else np.empty((0,), dtype=np.int8)
        obs_qual = None
        if fragment_obs_qual is not None:
            obs_qual = np.concatenate(qual_chunks, axis=0) if qual_chunks else np.empty((0,), dtype=np.uint8)
        return local_sample_offsets, centers, np.asarray(obs_offsets, dtype=np.int64), obs_pos, obs_code, obs_qual

    def _run_hmm_for_subset(
        self,
        *,
        ploidy: int,
        founder_panel: FounderPanel,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        other_count: np.ndarray,
        ref_weight: np.ndarray,
        alt_weight: np.ndarray,
        other_weight: np.ndarray,
        generations: np.ndarray,
        return_full_transition: bool,
        return_haplotype_posterior: bool,
        return_genotype_posterior: bool,
        fragment_sample_offsets: np.ndarray,
        fragment_center_idx: np.ndarray,
        fragment_obs_offsets: np.ndarray,
        fragment_obs_pos_idx: np.ndarray,
        fragment_obs_code: np.ndarray,
        fragment_obs_qual: np.ndarray | None,
    ):
        previous_mode = self.hmm.config.ploidy_mode
        previous_ploidy = self.hmm.config.ploidy
        try:
            self.hmm.config.ploidy = int(ploidy)
            self.hmm.config.ploidy_mode = "pseudo_haploid" if int(ploidy) == 1 else "diploid"
            return self.hmm.run(
                founder_panel=founder_panel,
                ref_count=ref_count,
                alt_count=alt_count,
                other_count=other_count,
                ref_weight=ref_weight,
                alt_weight=alt_weight,
                other_weight=other_weight,
                generations=generations,
                return_full_transition=return_full_transition,
                return_haplotype_posterior=return_haplotype_posterior,
                return_genotype_posterior=return_genotype_posterior,
                fragment_sample_offsets=fragment_sample_offsets,
                fragment_center_idx=fragment_center_idx,
                fragment_obs_offsets=fragment_obs_offsets,
                fragment_obs_pos_idx=fragment_obs_pos_idx,
                fragment_obs_code=fragment_obs_code,
                fragment_obs_qual=fragment_obs_qual,
            )
        finally:
            self.hmm.config.ploidy_mode = previous_mode
            self.hmm.config.ploidy = previous_ploidy

    def _run_blocks(
        self,
        samples: pd.DataFrame,
        positions_df: pd.DataFrame,
        founder_panel: FounderPanel,
        pedigree,
    ) -> None:
        if self.config.executor == "dask":
            self._run_blocks_dask(samples, positions_df, founder_panel, pedigree)
            return
        timings: list[dict[str, float | int]] = []
        generations = samples["generation"].to_numpy(dtype=np.float32, copy=False)
        sample_ids = samples["sample_id"].astype(str).to_numpy()
        sample_ploidy = (
            self._sample_ploidy.copy()
            if self._sample_ploidy is not None
            else np.full(len(samples), int(self.config.ploidy), dtype=np.int16)
        )
        has_bam_mask = samples["bam_path"].fillna("").astype(str).str.len().to_numpy(dtype=np.int32) > 0
        read_samples = samples.loc[has_bam_mask].reset_index(drop=True)
        if int(read_samples.shape[0]) > 0:
            self.read_extractor.open(read_samples)
        try:
            backend_autotuned = False
            for block in iter_position_blocks(positions_df, self.config.block_size):
                block_t0 = time.perf_counter()
                rss_block_start = _current_rss_mb() if self.config.profile_memory else None
                read_evidence = self.read_extractor.extract_block(read_samples, block) if int(read_samples.shape[0]) > 0 else None
                evidence = self._build_full_sample_evidence(
                    block_id=block.block_id,
                    chromosome=self.config.chromosome,
                    positions=block.dataframe["POS"].to_numpy(dtype=np.int64, copy=False),
                    ref=block.dataframe["REF"].astype(str).to_numpy(),
                    alt=block.dataframe["ALT"].astype(str).to_numpy(),
                    sample_ids=sample_ids,
                    has_bam_mask=has_bam_mask,
                    read_evidence=read_evidence,
                )
                try:
                    support_mask = evidence.depth > 0
                    if self._microarray_dosage is not None:
                        start = block.block_id * self.config.block_size
                        stop = min((block.block_id + 1) * self.config.block_size, self._microarray_dosage.shape[1])
                        self._inject_microarray_hard_calls(
                            evidence,
                            self._microarray_dosage[:, start:stop],
                            hard_call_weight=int(self.config.microarray_hard_call_weight),
                        )
                    t_after_reads = time.perf_counter()
                    rss_after_reads = _current_rss_mb() if self.config.profile_memory else None
                    if self.io_config.write_pileup:
                        write_parquet(
                            evidence.to_arrow(),
                            self.output_dir / "pileup" / f"block={block.block_id:06d}.parquet",
                            self.config.compression,
                            self.config.compression_level,
                        )

                    block_founders = founder_panel.slice(
                        block.block_id * self.config.block_size,
                        min((block.block_id + 1) * self.config.block_size, founder_panel.n_positions),
                    )
                    requested_genotype_outputs = (
                        self.io_config.write_genotype_posteriors
                        or self.io_config.write_genotype_calls
                    )
                    need_genotype_posterior = requested_genotype_outputs
                    positive_ploidies = np.unique(sample_ploidy[sample_ploidy > 0])
                    max_ploidy = int(np.max(sample_ploidy)) if sample_ploidy.size else int(self.config.ploidy)
                    if not backend_autotuned and np.any(sample_ploidy == 2):
                        dip_idx = np.flatnonzero(sample_ploidy == 2)
                        (
                            fso_dip,
                            fci_dip,
                            foo_dip,
                            fop_dip,
                            foc_dip,
                            foq_dip,
                        ) = self._subset_fragments_by_sample_indices(
                            dip_idx,
                            evidence.fragment_sample_offsets,
                            evidence.fragment_center_idx,
                            evidence.fragment_obs_offsets,
                            evidence.fragment_obs_pos_idx,
                            evidence.fragment_obs_code,
                            evidence.fragment_obs_qual,
                        )
                        self.hmm.config.ploidy_mode = "diploid"
                        self.hmm.autotune_backend(
                            founder_panel=block_founders,
                            ref_count=evidence.ref_count[dip_idx],
                            alt_count=evidence.alt_count[dip_idx],
                            generations=generations[dip_idx],
                            other_count=evidence.other_count[dip_idx],
                            ref_weight=evidence.ref_weight[dip_idx],
                            alt_weight=evidence.alt_weight[dip_idx],
                            other_weight=evidence.other_weight[dip_idx],
                            fragment_sample_offsets=fso_dip,
                            fragment_center_idx=fci_dip,
                            fragment_obs_offsets=foo_dip,
                            fragment_obs_pos_idx=fop_dip,
                            fragment_obs_code=foc_dip,
                            fragment_obs_qual=foq_dip,
                        )
                        backend_autotuned = True

                    return_full_transition = self.io_config.transition_output == "full"
                    return_haplotype = self.io_config.write_haplotype_probabilities
                    if np.all(sample_ploidy <= 0):
                        n_samples, n_positions = evidence.ref_count.shape
                        base_switch = self.hmm._switch_probabilities(
                            self.hmm.recombination_from_positions(block_founders.positions),
                            generations,
                        )
                        artifacts = HMMArtifacts(
                            dosage=np.full((n_samples, n_positions), np.nan, dtype=np.float32),
                            haplotype_posterior=(
                                np.zeros((n_samples, n_positions, block_founders.n_founders), dtype=np.float32)
                                if return_haplotype
                                else None
                            ),
                            genotype_posterior=None,
                            genotype_call=None,
                            recombination_rate=self.hmm.recombination_from_positions(block_founders.positions).astype(np.float32),
                            switch_probability=base_switch.astype(np.float32),
                            stay_probability=(1.0 - base_switch).astype(np.float32),
                            offdiag_probability=(base_switch / max(block_founders.n_founders - 1, 1)).astype(np.float32),
                            founder_alt_prob=block_founders.alt_prob.astype(np.float32, copy=True),
                            transition_probability=(
                                self.hmm._build_full_transitions(base_switch, block_founders.n_founders)
                                if return_full_transition
                                else None
                            ),
                        )
                    elif positive_ploidies.size == 1 and np.all(sample_ploidy > 0):
                        ploidy_i = int(positive_ploidies[0])
                        artifacts = self._run_hmm_for_subset(
                            ploidy=ploidy_i,
                            founder_panel=block_founders,
                            ref_count=evidence.ref_count,
                            alt_count=evidence.alt_count,
                            other_count=evidence.other_count,
                            ref_weight=evidence.ref_weight,
                            alt_weight=evidence.alt_weight,
                            other_weight=evidence.other_weight,
                            generations=generations,
                            return_full_transition=return_full_transition,
                            return_haplotype_posterior=return_haplotype,
                            return_genotype_posterior=need_genotype_posterior,
                            fragment_sample_offsets=evidence.fragment_sample_offsets,
                            fragment_center_idx=evidence.fragment_center_idx,
                            fragment_obs_offsets=evidence.fragment_obs_offsets,
                            fragment_obs_pos_idx=evidence.fragment_obs_pos_idx,
                            fragment_obs_code=evidence.fragment_obs_code,
                            fragment_obs_qual=evidence.fragment_obs_qual,
                        )
                    else:
                        n_samples, n_positions = evidence.ref_count.shape
                        dosage_out = np.full((n_samples, n_positions), np.nan, dtype=np.float32)
                        hap_out = (
                            np.zeros((n_samples, n_positions, block_founders.n_founders), dtype=np.float32)
                            if return_haplotype
                            else None
                        )
                        gp_classes = max(max_ploidy + 1, 1)
                        gp_out = np.zeros((n_samples, n_positions, gp_classes), dtype=np.float32) if need_genotype_posterior else None
                        switch_out = np.zeros((n_samples, n_positions), dtype=np.float32)
                        stay_out = np.zeros((n_samples, n_positions), dtype=np.float32)
                        offdiag_out = np.zeros((n_samples, n_positions), dtype=np.float32)
                        founder_acc = np.zeros_like(block_founders.alt_prob, dtype=np.float32)
                        founder_w = 0.0

                        for ploidy_i in positive_ploidies.tolist():
                            mask = sample_ploidy == int(ploidy_i)
                            idx = np.flatnonzero(mask)
                            if idx.size == 0:
                                continue
                            (
                                fso,
                                fci,
                                foo,
                                fop,
                                foc,
                                foq,
                            ) = self._subset_fragments_by_sample_indices(
                                idx,
                                evidence.fragment_sample_offsets,
                                evidence.fragment_center_idx,
                                evidence.fragment_obs_offsets,
                                evidence.fragment_obs_pos_idx,
                                evidence.fragment_obs_code,
                                evidence.fragment_obs_qual,
                            )
                            sub = self._run_hmm_for_subset(
                                ploidy=int(ploidy_i),
                                founder_panel=block_founders,
                                ref_count=evidence.ref_count[idx],
                                alt_count=evidence.alt_count[idx],
                                other_count=evidence.other_count[idx],
                                ref_weight=evidence.ref_weight[idx],
                                alt_weight=evidence.alt_weight[idx],
                                other_weight=evidence.other_weight[idx],
                                generations=generations[idx],
                                return_full_transition=return_full_transition,
                                return_haplotype_posterior=return_haplotype,
                                return_genotype_posterior=need_genotype_posterior,
                                fragment_sample_offsets=fso,
                                fragment_center_idx=fci,
                                fragment_obs_offsets=foo,
                                fragment_obs_pos_idx=fop,
                                fragment_obs_code=foc,
                                fragment_obs_qual=foq,
                            )
                            dosage_out[idx] = sub.dosage.astype(np.float32, copy=False)
                            if hap_out is not None and sub.haplotype_posterior is not None:
                                hap_out[idx] = sub.haplotype_posterior
                            if gp_out is not None and sub.genotype_posterior is not None:
                                n_cls = min(gp_out.shape[2], sub.genotype_posterior.shape[2])
                                gp_out[idx, :, :n_cls] = sub.genotype_posterior[:, :, :n_cls]
                            switch_out[idx] = sub.switch_probability
                            stay_out[idx] = sub.stay_probability
                            offdiag_out[idx] = sub.offdiag_probability
                            founder_acc += sub.founder_alt_prob.astype(np.float32, copy=False) * float(idx.size)
                            founder_w += float(idx.size)

                        if np.any(sample_ploidy == 0):
                            zero_idx = np.flatnonzero(sample_ploidy == 0)
                            dosage_out[zero_idx] = np.nan
                            if gp_out is not None:
                                gp_out[zero_idx] = np.nan

                        founder_final = (
                            founder_acc / max(founder_w, 1.0)
                            if founder_w > 0.0
                            else block_founders.alt_prob.astype(np.float32, copy=True)
                        )
                        gt_out = None
                        if gp_out is not None:
                            gt_out = np.argmax(np.nan_to_num(gp_out, nan=-1.0), axis=2).astype(np.int8, copy=False)
                            gt_out[~np.isfinite(gp_out).any(axis=2)] = -1
                        artifacts = HMMArtifacts(
                            dosage=dosage_out,
                            haplotype_posterior=hap_out,
                            genotype_posterior=gp_out,
                            genotype_call=gt_out,
                            recombination_rate=self.hmm.recombination_from_positions(block_founders.positions).astype(np.float32),
                            switch_probability=switch_out,
                            stay_probability=stay_out,
                            offdiag_probability=offdiag_out,
                            founder_alt_prob=founder_final.astype(np.float32, copy=False),
                            transition_probability=None,
                        )
                    t_after_hmm = time.perf_counter()
                    rss_after_hmm = _current_rss_mb() if self.config.profile_memory else None
                    dosage = smooth_dosage_with_pedigree(
                        artifacts.dosage,
                        pedigree=pedigree,
                        strength=self.config.pedigree_strength,
                    )
                    calibrated_gp = None
                    calibrated_gt = None
                    call_correct_probability = None
                    calibration_meta: dict[str, object] = {"status": "disabled"}
                    if need_genotype_posterior:
                        if self.config.calibrate_genotype_posteriors:
                            calibrated_gp = calibrate_genotype_posterior(
                                artifacts.genotype_posterior,
                                dosage=dosage,
                                depth=evidence.depth.astype(np.float32, copy=False),
                                temperature=self.config.genotype_posterior_temperature,
                                blend=self.config.genotype_posterior_blend,
                                ploidy=max_ploidy,
                            )
                        else:
                            calibrated_gp = artifacts.genotype_posterior
                        if calibrated_gp is None:
                            calibrated_gp = calibrate_genotype_posterior(
                                None,
                                dosage=dosage,
                                depth=evidence.depth.astype(np.float32, copy=False),
                                temperature=self.config.genotype_posterior_temperature,
                                blend=self.config.genotype_posterior_blend,
                                ploidy=max_ploidy,
                            )
                        if self.config.use_lightgbm_calibrator and calibrated_gp is not None and calibrated_gp.shape[2] == 3:
                            truth_gt_block = None
                            if self._microarray_dosage is not None:
                                start = block.block_id * self.config.block_size
                                stop = min((block.block_id + 1) * self.config.block_size, self._microarray_dosage.shape[1])
                                micro_block = self._microarray_dosage[:, start:stop]
                                truth_gt_block = np.full(micro_block.shape, -1, dtype=np.int8)
                                valid_micro = np.isfinite(micro_block)
                                if np.any(valid_micro):
                                    truth_gt_block[valid_micro] = np.clip(
                                        np.rint(micro_block[valid_micro]),
                                        0,
                                        2,
                                    ).astype(np.int8, copy=False)
                            if truth_gt_block is not None and int(np.sum(truth_gt_block >= 0)) >= 128:
                                calibrated_gp, call_correct_probability, calibration_meta = calibrate_genotype_posterior_full_stack(
                                    raw_posterior=calibrated_gp,
                                    dosage=dosage,
                                    truth_genotype=truth_gt_block,
                                    depth=evidence.depth.astype(np.float32, copy=False),
                                    ref_count=evidence.ref_count.astype(np.float32, copy=False),
                                    alt_count=evidence.alt_count.astype(np.float32, copy=False),
                                    other_count=evidence.other_count.astype(np.float32, copy=False),
                                    support_mask=support_mask.astype(np.float32, copy=False),
                                    generations=generations.astype(np.float32, copy=False),
                                    samples_df=samples,
                                    train_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                                    predict_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                                    window=int(self.config.calibration_context_window),
                                    block_size=int(self.config.calibration_block_snps),
                                    max_train_rows=int(self.config.calibration_max_train_rows),
                                    use_optuna=bool(self.config.calibration_use_optuna),
                                    optuna_trials=int(self.config.calibration_optuna_trials),
                                    seed=int(self.config.random_seed + block.block_id),
                                    class_weight_mode="balanced",
                                    apply_isotonic=True,
                                )
                            elif truth_gt_block is not None:
                                calibrated_gp, calibration_meta = calibrate_genotype_posterior_block_context(
                                    raw_posterior=calibrated_gp,
                                    dosage=dosage,
                                    truth_genotype=truth_gt_block,
                                    samples_df=samples,
                                    train_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                                    predict_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                                    window=int(self.config.calibration_context_window),
                                    block_size=int(self.config.calibration_block_snps),
                                    max_train_rows=int(self.config.calibration_max_train_rows),
                                    use_optuna=bool(self.config.calibration_use_optuna),
                                    optuna_trials=int(self.config.calibration_optuna_trials),
                                    seed=int(self.config.random_seed + block.block_id),
                                )
                        calibrated_gp = self._mask_genotype_posterior_by_ploidy(calibrated_gp, sample_ploidy)
                        stitch_threshold = None
                        min_confidence = 0.0
                        min_margin = 0.0
                        call_correct_threshold = 0.0
                        if self.config.genotype_call_mode == "stitch_no_call":
                            stitch_threshold = float(self.config.genotype_call_stitch_threshold)
                            min_margin = float(self.config.genotype_call_min_margin)
                        elif self.config.genotype_call_mode == "quality_gated":
                            min_confidence = float(self.config.genotype_call_min_confidence)
                            min_margin = float(self.config.genotype_call_min_margin)
                            call_correct_threshold = float(self.config.genotype_call_correctness_threshold)
                            if call_correct_threshold <= 0.0 and isinstance(calibration_meta, dict):
                                cmeta = calibration_meta.get("call_correctness")
                                if isinstance(cmeta, dict):
                                    cthr = cmeta.get("threshold")
                                    if cthr is not None:
                                        call_correct_threshold = float(cthr)
                        calibrated_gt = genotype_call_from_posterior(
                            calibrated_gp,
                            min_confidence=min_confidence,
                            min_margin=min_margin,
                            stitch_gp_threshold=stitch_threshold,
                            call_correct_probability=call_correct_probability,
                            call_correct_threshold=call_correct_threshold,
                        )
                    elif requested_genotype_outputs:
                        calibrated_gp = dosage_to_genotype_posterior(
                            np.clip(dosage, 0.0, max(max_ploidy, 0)).astype(np.float32, copy=False),
                            temperature=self.config.genotype_posterior_temperature,
                            ploidy=max_ploidy,
                        )
                        calibrated_gp = self._mask_genotype_posterior_by_ploidy(calibrated_gp, sample_ploidy)
                        max_gt = np.broadcast_to(sample_ploidy[:, None], dosage.shape)
                        gt = np.full(dosage.shape, -1, dtype=np.int8)
                        valid = np.isfinite(dosage) & (max_gt > 0)
                        if np.any(valid):
                            rounded = np.rint(dosage).astype(np.float32, copy=False)
                            clipped = np.clip(rounded, 0.0, max_gt.astype(np.float32, copy=False))
                            gt[valid] = clipped[valid].astype(np.int8, copy=False)
                        calibrated_gt = gt
                    if calibrated_gt is not None:
                        valid_call = np.isfinite(dosage) & (sample_ploidy[:, None] > 0)
                        calibrated_gt = np.where(valid_call, calibrated_gt, -1).astype(np.int8, copy=False)
                    t_after_calibration = time.perf_counter()
                    rss_after_calibration = _current_rss_mb() if self.config.profile_memory else None
                    self._write_hmm_outputs(
                        sample_ids,
                        block,
                        dosage,
                        artifacts,
                        support_mask=support_mask,
                        genotype_posterior=calibrated_gp,
                        genotype_call=calibrated_gt,
                    )
                    t_after_write = time.perf_counter()
                    rss_after_write = _current_rss_mb() if self.config.profile_memory else None
                    block_timing = {
                        "block_id": block.block_id,
                        "seconds_read_extract": t_after_reads - block_t0,
                        "seconds_hmm": t_after_hmm - t_after_reads,
                        "seconds_calibration": t_after_calibration - t_after_hmm,
                        "seconds_write": t_after_write - t_after_calibration,
                        "seconds_total": t_after_write - block_t0,
                        "mean_depth": float(np.mean(evidence.depth)),
                        "n_reads": int(np.sum(evidence.n_overlapping_reads)),
                    }
                    if calibrated_gt is not None:
                        block_timing["call_rate"] = float(np.mean(calibrated_gt >= 0))
                        block_timing["no_call_rate"] = float(np.mean(calibrated_gt < 0))
                    if isinstance(calibration_meta, dict):
                        cc = calibration_meta.get("call_correctness")
                        if isinstance(cc, dict):
                            thr = cc.get("threshold")
                            if thr is not None:
                                block_timing["call_correctness_threshold"] = float(thr)
                    if self.config.profile_memory:
                        block_timing.update(
                            {
                                "rss_mb_block_start": float(rss_block_start or 0.0),
                                "rss_mb_after_reads": float(rss_after_reads or 0.0),
                                "rss_mb_after_hmm": float(rss_after_hmm or 0.0),
                                "rss_mb_after_calibration": float(rss_after_calibration or 0.0),
                                "rss_mb_after_write": float(rss_after_write or 0.0),
                            }
                        )
                    timings.append(block_timing)
                finally:
                    if read_evidence is not None:
                        read_evidence.release()
                    evidence.release()
                    if self.config.gc_collect_every_block:
                        gc.collect()
        finally:
            if int(read_samples.shape[0]) > 0:
                self.read_extractor.close()

        timings_path = self.output_dir / "stage_timings.json"
        timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
        if self.config.profile_memory and timings:
            peak_rss = max(float(row.get("rss_mb_after_write", 0.0)) for row in timings)
            peak_hmm_rss = max(float(row.get("rss_mb_after_hmm", 0.0)) for row in timings)
            mem_summary = {
                "peak_rss_mb": peak_rss,
                "peak_hmm_rss_mb": peak_hmm_rss,
                "blocks_profiled": len(timings),
            }
            (self.output_dir / "memory_profile_summary.json").write_text(
                json.dumps(mem_summary, indent=2),
                encoding="utf-8",
            )

    def _missing_artifacts(
        self,
        *,
        n_samples: int,
        n_positions: int,
        generations: np.ndarray,
        block_founders: FounderPanel,
        return_haplotype: bool,
        return_full_transition: bool,
    ) -> HMMArtifacts:
        base_switch = self.hmm._switch_probabilities(
            self.hmm.recombination_from_positions(block_founders.positions),
            generations,
        )
        return HMMArtifacts(
            dosage=np.full((n_samples, n_positions), np.nan, dtype=np.float32),
            haplotype_posterior=(
                np.zeros((n_samples, n_positions, block_founders.n_founders), dtype=np.float32)
                if return_haplotype
                else None
            ),
            genotype_posterior=None,
            genotype_call=None,
            recombination_rate=self.hmm.recombination_from_positions(block_founders.positions).astype(np.float32),
            switch_probability=base_switch.astype(np.float32),
            stay_probability=(1.0 - base_switch).astype(np.float32),
            offdiag_probability=(base_switch / max(block_founders.n_founders - 1, 1)).astype(np.float32),
            founder_alt_prob=block_founders.alt_prob.astype(np.float32, copy=True),
            transition_probability=(
                self.hmm._build_full_transitions(base_switch, block_founders.n_founders)
                if return_full_transition
                else None
            ),
        )

    def _merge_dask_hmm_task_results(
        self,
        *,
        results: list[DaskHMMTaskResult],
        n_samples: int,
        n_positions: int,
        sample_ploidy: np.ndarray,
        max_ploidy: int,
        block_founders: FounderPanel,
        return_haplotype: bool,
        return_genotype_posterior: bool,
    ) -> HMMArtifacts:
        dosage_out = np.full((n_samples, n_positions), np.nan, dtype=np.float32)
        hap_out = (
            np.zeros((n_samples, n_positions, block_founders.n_founders), dtype=np.float32)
            if return_haplotype
            else None
        )
        gp_classes = max(max_ploidy + 1, 1)
        gp_out = np.zeros((n_samples, n_positions, gp_classes), dtype=np.float32) if return_genotype_posterior else None
        switch_out = np.zeros((n_samples, n_positions), dtype=np.float32)
        stay_out = np.zeros((n_samples, n_positions), dtype=np.float32)
        offdiag_out = np.zeros((n_samples, n_positions), dtype=np.float32)
        founder_acc = np.zeros_like(block_founders.alt_prob, dtype=np.float32)
        founder_w = 0.0
        transition_probability = None

        for result in results:
            idx = result.sample_indices.astype(np.int64, copy=False)
            sub = result.artifacts
            dosage_out[idx] = sub.dosage.astype(np.float32, copy=False)
            if hap_out is not None and sub.haplotype_posterior is not None:
                hap_out[idx] = sub.haplotype_posterior
            if gp_out is not None and sub.genotype_posterior is not None:
                n_cls = min(gp_out.shape[2], sub.genotype_posterior.shape[2])
                gp_out[idx, :, :n_cls] = sub.genotype_posterior[:, :, :n_cls]
            switch_out[idx] = sub.switch_probability
            stay_out[idx] = sub.stay_probability
            offdiag_out[idx] = sub.offdiag_probability
            founder_acc += sub.founder_alt_prob.astype(np.float32, copy=False) * float(idx.size)
            founder_w += float(idx.size)
            if (
                len(results) == 1
                and idx.size == n_samples
                and np.array_equal(idx, np.arange(n_samples, dtype=np.int64))
            ):
                transition_probability = sub.transition_probability

        if np.any(sample_ploidy == 0):
            zero_idx = np.flatnonzero(sample_ploidy == 0)
            dosage_out[zero_idx] = np.nan
            if gp_out is not None:
                gp_out[zero_idx] = np.nan

        founder_final = (
            founder_acc / max(founder_w, 1.0)
            if founder_w > 0.0
            else block_founders.alt_prob.astype(np.float32, copy=True)
        )
        gt_out = None
        if gp_out is not None:
            gt_out = np.argmax(np.nan_to_num(gp_out, nan=-1.0), axis=2).astype(np.int8, copy=False)
            gt_out[~np.isfinite(gp_out).any(axis=2)] = -1
        return HMMArtifacts(
            dosage=dosage_out,
            haplotype_posterior=hap_out,
            genotype_posterior=gp_out,
            genotype_call=gt_out,
            recombination_rate=self.hmm.recombination_from_positions(block_founders.positions).astype(np.float32),
            switch_probability=switch_out,
            stay_probability=stay_out,
            offdiag_probability=offdiag_out,
            founder_alt_prob=founder_final.astype(np.float32, copy=False),
            transition_probability=transition_probability,
        )

    def _finalize_and_write_block(
        self,
        *,
        samples: pd.DataFrame,
        sample_ids: np.ndarray,
        sample_ploidy: np.ndarray,
        generations: np.ndarray,
        pedigree,
        block,
        block_start_offset: int,
        evidence: ReadEvidenceBlock,
        support_mask: np.ndarray,
        artifacts: HMMArtifacts,
        need_genotype_posterior: bool,
        requested_genotype_outputs: bool,
        max_ploidy: int,
        block_t0: float,
        t_after_reads: float,
        t_after_hmm: float,
        rss_block_start: float | None,
        rss_after_reads: float | None,
        dask_extra: dict[str, float | int | str | bool] | None = None,
    ) -> dict[str, float | int | str | bool]:
        rss_after_hmm = _current_rss_mb() if self.config.profile_memory else None
        dosage = smooth_dosage_with_pedigree(
            artifacts.dosage,
            pedigree=pedigree,
            strength=self.config.pedigree_strength,
        )
        calibrated_gp = None
        calibrated_gt = None
        call_correct_probability = None
        calibration_meta: dict[str, object] = {"status": "disabled"}
        if need_genotype_posterior:
            if self.config.calibrate_genotype_posteriors:
                calibrated_gp = calibrate_genotype_posterior(
                    artifacts.genotype_posterior,
                    dosage=dosage,
                    depth=evidence.depth.astype(np.float32, copy=False),
                    temperature=self.config.genotype_posterior_temperature,
                    blend=self.config.genotype_posterior_blend,
                    ploidy=max_ploidy,
                )
            else:
                calibrated_gp = artifacts.genotype_posterior
            if calibrated_gp is None:
                calibrated_gp = calibrate_genotype_posterior(
                    None,
                    dosage=dosage,
                    depth=evidence.depth.astype(np.float32, copy=False),
                    temperature=self.config.genotype_posterior_temperature,
                    blend=self.config.genotype_posterior_blend,
                    ploidy=max_ploidy,
                )
            if self.config.use_lightgbm_calibrator and calibrated_gp is not None and calibrated_gp.shape[2] == 3:
                truth_gt_block = None
                if self._microarray_dosage is not None:
                    start = int(block_start_offset)
                    stop = min(start + int(evidence.ref_count.shape[1]), self._microarray_dosage.shape[1])
                    micro_block = self._microarray_dosage[:, start:stop]
                    truth_gt_block = np.full(micro_block.shape, -1, dtype=np.int8)
                    valid_micro = np.isfinite(micro_block)
                    if np.any(valid_micro):
                        truth_gt_block[valid_micro] = np.clip(
                            np.rint(micro_block[valid_micro]),
                            0,
                            2,
                        ).astype(np.int8, copy=False)
                if truth_gt_block is not None and int(np.sum(truth_gt_block >= 0)) >= 128:
                    calibrated_gp, call_correct_probability, calibration_meta = calibrate_genotype_posterior_full_stack(
                        raw_posterior=calibrated_gp,
                        dosage=dosage,
                        truth_genotype=truth_gt_block,
                        depth=evidence.depth.astype(np.float32, copy=False),
                        ref_count=evidence.ref_count.astype(np.float32, copy=False),
                        alt_count=evidence.alt_count.astype(np.float32, copy=False),
                        other_count=evidence.other_count.astype(np.float32, copy=False),
                        support_mask=support_mask.astype(np.float32, copy=False),
                        generations=generations.astype(np.float32, copy=False),
                        samples_df=samples,
                        train_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                        predict_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                        window=int(self.config.calibration_context_window),
                        block_size=int(self.config.calibration_block_snps),
                        max_train_rows=int(self.config.calibration_max_train_rows),
                        use_optuna=bool(self.config.calibration_use_optuna),
                        optuna_trials=int(self.config.calibration_optuna_trials),
                        seed=int(self.config.random_seed + block.block_id),
                        class_weight_mode="balanced",
                        apply_isotonic=True,
                    )
                elif truth_gt_block is not None:
                    calibrated_gp, calibration_meta = calibrate_genotype_posterior_block_context(
                        raw_posterior=calibrated_gp,
                        dosage=dosage,
                        truth_genotype=truth_gt_block,
                        samples_df=samples,
                        train_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                        predict_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                        window=int(self.config.calibration_context_window),
                        block_size=int(self.config.calibration_block_snps),
                        max_train_rows=int(self.config.calibration_max_train_rows),
                        use_optuna=bool(self.config.calibration_use_optuna),
                        optuna_trials=int(self.config.calibration_optuna_trials),
                        seed=int(self.config.random_seed + block.block_id),
                    )
            calibrated_gp = self._mask_genotype_posterior_by_ploidy(calibrated_gp, sample_ploidy)
            stitch_threshold = None
            min_confidence = 0.0
            min_margin = 0.0
            call_correct_threshold = 0.0
            if self.config.genotype_call_mode == "stitch_no_call":
                stitch_threshold = float(self.config.genotype_call_stitch_threshold)
                min_margin = float(self.config.genotype_call_min_margin)
            elif self.config.genotype_call_mode == "quality_gated":
                min_confidence = float(self.config.genotype_call_min_confidence)
                min_margin = float(self.config.genotype_call_min_margin)
                call_correct_threshold = float(self.config.genotype_call_correctness_threshold)
                if call_correct_threshold <= 0.0 and isinstance(calibration_meta, dict):
                    cmeta = calibration_meta.get("call_correctness")
                    if isinstance(cmeta, dict):
                        cthr = cmeta.get("threshold")
                        if cthr is not None:
                            call_correct_threshold = float(cthr)
            calibrated_gt = genotype_call_from_posterior(
                calibrated_gp,
                min_confidence=min_confidence,
                min_margin=min_margin,
                stitch_gp_threshold=stitch_threshold,
                call_correct_probability=call_correct_probability,
                call_correct_threshold=call_correct_threshold,
            )
        elif requested_genotype_outputs:
            calibrated_gp = dosage_to_genotype_posterior(
                np.clip(dosage, 0.0, max(max_ploidy, 0)).astype(np.float32, copy=False),
                temperature=self.config.genotype_posterior_temperature,
                ploidy=max_ploidy,
            )
            calibrated_gp = self._mask_genotype_posterior_by_ploidy(calibrated_gp, sample_ploidy)
            max_gt = np.broadcast_to(sample_ploidy[:, None], dosage.shape)
            gt = np.full(dosage.shape, -1, dtype=np.int8)
            valid = np.isfinite(dosage) & (max_gt > 0)
            if np.any(valid):
                rounded = np.rint(dosage).astype(np.float32, copy=False)
                clipped = np.clip(rounded, 0.0, max_gt.astype(np.float32, copy=False))
                gt[valid] = clipped[valid].astype(np.int8, copy=False)
            calibrated_gt = gt
        if calibrated_gt is not None:
            valid_call = np.isfinite(dosage) & (sample_ploidy[:, None] > 0)
            calibrated_gt = np.where(valid_call, calibrated_gt, -1).astype(np.int8, copy=False)
        t_after_calibration = time.perf_counter()
        rss_after_calibration = _current_rss_mb() if self.config.profile_memory else None
        self._write_hmm_outputs(
            sample_ids,
            block,
            dosage,
            artifacts,
            support_mask=support_mask,
            genotype_posterior=calibrated_gp,
            genotype_call=calibrated_gt,
        )
        t_after_write = time.perf_counter()
        rss_after_write = _current_rss_mb() if self.config.profile_memory else None
        block_timing: dict[str, float | int | str | bool] = {
            "block_id": block.block_id,
            "seconds_read_extract": t_after_reads - block_t0,
            "seconds_hmm": t_after_hmm - t_after_reads,
            "seconds_calibration": t_after_calibration - t_after_hmm,
            "seconds_write": t_after_write - t_after_calibration,
            "seconds_total": t_after_write - block_t0,
            "mean_depth": float(np.mean(evidence.depth)),
            "n_reads": int(np.sum(evidence.n_overlapping_reads)),
        }
        if calibrated_gt is not None:
            block_timing["call_rate"] = float(np.mean(calibrated_gt >= 0))
            block_timing["no_call_rate"] = float(np.mean(calibrated_gt < 0))
        if isinstance(calibration_meta, dict):
            cc = calibration_meta.get("call_correctness")
            if isinstance(cc, dict):
                thr = cc.get("threshold")
                if thr is not None:
                    block_timing["call_correctness_threshold"] = float(thr)
        if dask_extra:
            block_timing.update(dask_extra)
        if self.config.profile_memory:
            block_timing.update(
                {
                    "rss_mb_block_start": float(rss_block_start or 0.0),
                    "rss_mb_after_reads": float(rss_after_reads or 0.0),
                    "rss_mb_after_hmm": float(rss_after_hmm or 0.0),
                    "rss_mb_after_calibration": float(rss_after_calibration or 0.0),
                    "rss_mb_after_write": float(rss_after_write or 0.0),
                }
            )
        return block_timing

    def _run_blocks_dask(
        self,
        samples: pd.DataFrame,
        positions_df: pd.DataFrame,
        founder_panel: FounderPanel,
        pedigree,
    ) -> None:
        try:
            from dask import delayed
            from dask.distributed import Client, LocalCluster, get_task_stream, performance_report
        except ImportError as exc:  # pragma: no cover - depends on optional runtime packaging.
            raise ImportError(
                "The Dask executor requires dask.distributed. Install stitchv2 with the 'distributed' dependency."
            ) from exc

        if self.config.dask_scheduler != "local":
            raise ValueError("Only --dask-scheduler local is currently implemented.")

        timings: list[dict[str, float | int | str | bool]] = []
        generations = samples["generation"].to_numpy(dtype=np.float32, copy=False)
        sample_ids = samples["sample_id"].astype(str).to_numpy()
        sample_ploidy = (
            self._sample_ploidy.copy()
            if self._sample_ploidy is not None
            else np.full(len(samples), int(self.config.ploidy), dtype=np.int16)
        )
        max_ploidy_global = int(np.max(sample_ploidy)) if sample_ploidy.size else int(self.config.ploidy)
        requested_genotype_outputs = self.io_config.write_genotype_posteriors or self.io_config.write_genotype_calls
        need_genotype_posterior = requested_genotype_outputs
        return_full_transition = self.io_config.transition_output == "full"
        return_haplotype = self.io_config.write_haplotype_probabilities
        configured_sample_batch = int(self.config.dask_sample_batch_size) or int(self.config.jax_sample_batch_size)
        chunk_plan = plan_dask_chunks(
            n_samples=len(samples),
            n_variants=len(positions_df),
            n_founders=founder_panel.n_founders,
            max_ploidy=max(max_ploidy_global, 1),
            configured_block_size=self.config.block_size,
            configured_sample_batch_size=configured_sample_batch,
            target_task_memory_mb=self.config.dask_target_task_memory_mb,
            min_block_size=self.config.dask_min_block_size,
            min_sample_batch_size=self.config.dask_min_sample_batch_size,
            return_genotype_posterior=need_genotype_posterior,
            return_haplotype_posterior=return_haplotype,
            return_full_transition=return_full_transition,
            force_generic_ploidy_hmm=self.config.force_generic_ploidy_hmm,
        )
        effective_block_size = int(chunk_plan.block_size)
        sample_batch_size = int(chunk_plan.sample_batch_size)

        n_workers = int(self.config.dask_n_workers)
        if n_workers <= 0:
            n_workers = max(1, min(os.cpu_count() or 1, 4))
        dashboard_address = self.config.dask_dashboard_address
        memory_limit = self.config.dask_memory_limit or "auto"
        cluster_kwargs = {
            "n_workers": n_workers,
            "threads_per_worker": max(int(self.config.dask_threads_per_worker), 1),
            "processes": bool(self.config.dask_processes),
            "memory_limit": memory_limit,
            "dashboard_address": dashboard_address,
        }
        if isinstance(dashboard_address, str) and dashboard_address.startswith("127.0.0.1:"):
            cluster_kwargs["host"] = "127.0.0.1"
        if not bool(self.config.dask_processes):
            cluster_kwargs["protocol"] = "inproc://"
        cluster = LocalCluster(**cluster_kwargs)
        client = Client(cluster)
        dashboard_link = str(client.dashboard_link) if client.dashboard_link else None
        if dashboard_link:
            print(f"Dask dashboard: {dashboard_link}", flush=True)

        has_bam_mask = samples["bam_path"].fillna("").astype(str).str.len().to_numpy(dtype=np.int32) > 0
        read_samples = samples.loc[has_bam_mask].reset_index(drop=True)
        if int(read_samples.shape[0]) > 0:
            self.read_extractor.open(read_samples)

        performance_report_path = Path(self.config.dask_performance_report) if self.config.dask_performance_report else None
        task_stream_path = Path(self.config.dask_task_stream) if self.config.dask_task_stream else None
        if performance_report_path is not None and not performance_report_path.is_absolute():
            performance_report_path = self.output_dir / performance_report_path
        if task_stream_path is not None and not task_stream_path.is_absolute():
            task_stream_path = self.output_dir / task_stream_path
        task_stream = None
        total_tasks = 0
        task_diagnostics: list[dict[str, object]] = []
        mutable_founder_forced_batches = False
        try:
            with ExitStack() as stack:
                if performance_report_path is not None:
                    performance_report_path.parent.mkdir(parents=True, exist_ok=True)
                    stack.enter_context(performance_report(filename=str(performance_report_path)))
                if task_stream_path is not None:
                    task_stream = stack.enter_context(get_task_stream(client=client, plot=False))

                for block in iter_position_blocks(positions_df, effective_block_size):
                    block_t0 = time.perf_counter()
                    block_start_offset = int(block.block_id * effective_block_size)
                    rss_block_start = _current_rss_mb() if self.config.profile_memory else None
                    read_evidence = self.read_extractor.extract_block(read_samples, block) if int(read_samples.shape[0]) > 0 else None
                    evidence = self._build_full_sample_evidence(
                        block_id=block.block_id,
                        chromosome=self.config.chromosome,
                        positions=block.dataframe["POS"].to_numpy(dtype=np.int64, copy=False),
                        ref=block.dataframe["REF"].astype(str).to_numpy(),
                        alt=block.dataframe["ALT"].astype(str).to_numpy(),
                        sample_ids=sample_ids,
                        has_bam_mask=has_bam_mask,
                        read_evidence=read_evidence,
                    )
                    try:
                        support_mask = evidence.depth > 0
                        if self._microarray_dosage is not None:
                            start = block_start_offset
                            stop = min(start + int(evidence.ref_count.shape[1]), self._microarray_dosage.shape[1])
                            self._inject_microarray_hard_calls(
                                evidence,
                                self._microarray_dosage[:, start:stop],
                                hard_call_weight=int(self.config.microarray_hard_call_weight),
                            )
                        t_after_reads = time.perf_counter()
                        rss_after_reads = _current_rss_mb() if self.config.profile_memory else None
                        if self.io_config.write_pileup:
                            write_parquet(
                                evidence.to_arrow(),
                                self.output_dir / "pileup" / f"block={block.block_id:06d}.parquet",
                                self.config.compression,
                                self.config.compression_level,
                            )

                        block_founders = founder_panel.slice(
                            block_start_offset,
                            min(block_start_offset + int(evidence.ref_count.shape[1]), founder_panel.n_positions),
                        )
                        positive_ploidies = np.unique(sample_ploidy[sample_ploidy > 0])
                        max_ploidy = int(np.max(sample_ploidy)) if sample_ploidy.size else int(self.config.ploidy)
                        n_samples, n_positions = evidence.ref_count.shape
                        task_results: list[DaskHMMTaskResult] = []

                        if np.all(sample_ploidy <= 0):
                            artifacts = self._missing_artifacts(
                                n_samples=n_samples,
                                n_positions=n_positions,
                                generations=generations,
                                block_founders=block_founders,
                                return_haplotype=return_haplotype,
                                return_full_transition=return_full_transition,
                            )
                        else:
                            tasks = []
                            force_full_group = not bool(np.all(block_founders.immutable_mask))
                            mutable_founder_forced_batches = mutable_founder_forced_batches or force_full_group
                            hmm_config = self.config.hmm()
                            for ploidy_i in positive_ploidies.tolist():
                                idx_all = np.flatnonzero(sample_ploidy == int(ploidy_i))
                                if idx_all.size == 0:
                                    continue
                                if force_full_group:
                                    batches = [idx_all]
                                else:
                                    batches = [
                                        idx_all[start : start + sample_batch_size]
                                        for start in range(0, idx_all.size, sample_batch_size)
                                    ]
                                for idx in batches:
                                    (
                                        fso,
                                        fci,
                                        foo,
                                        fop,
                                        foc,
                                        foq,
                                    ) = self._subset_fragments_by_sample_indices(
                                        idx,
                                        evidence.fragment_sample_offsets,
                                        evidence.fragment_center_idx,
                                        evidence.fragment_obs_offsets,
                                        evidence.fragment_obs_pos_idx,
                                        evidence.fragment_obs_code,
                                        evidence.fragment_obs_qual,
                                    )
                                    tasks.append(
                                        delayed(run_hmm_leaf_task, pure=False)(
                                            block_id=block.block_id,
                                            sample_indices=idx.astype(np.int64, copy=True),
                                            ploidy=int(ploidy_i),
                                            hmm_config=hmm_config,
                                            founder_panel=block_founders,
                                            ref_count=evidence.ref_count[idx].copy(),
                                            alt_count=evidence.alt_count[idx].copy(),
                                            generations=generations[idx].copy(),
                                            other_count=evidence.other_count[idx].copy(),
                                            ref_weight=evidence.ref_weight[idx].copy(),
                                            alt_weight=evidence.alt_weight[idx].copy(),
                                            other_weight=evidence.other_weight[idx].copy(),
                                            return_full_transition=return_full_transition and len(batches) == 1 and idx.size == n_samples,
                                            return_haplotype_posterior=return_haplotype,
                                            return_genotype_posterior=need_genotype_posterior,
                                            fragment_sample_offsets=fso,
                                            fragment_center_idx=fci,
                                            fragment_obs_offsets=foo,
                                            fragment_obs_pos_idx=fop,
                                            fragment_obs_code=foc,
                                            fragment_obs_qual=foq,
                                        )
                                    )

                            futures = client.compute(tasks)
                            task_results = list(client.gather(futures))
                            total_tasks += len(task_results)
                            task_diagnostics.extend(result.diagnostics() for result in task_results)
                            if (
                                len(task_results) == 1
                                and np.array_equal(task_results[0].sample_indices, np.arange(n_samples, dtype=np.int64))
                                and not np.any(sample_ploidy == 0)
                            ):
                                artifacts = task_results[0].artifacts
                            else:
                                artifacts = self._merge_dask_hmm_task_results(
                                    results=task_results,
                                    n_samples=n_samples,
                                    n_positions=n_positions,
                                    sample_ploidy=sample_ploidy,
                                    max_ploidy=max_ploidy,
                                    block_founders=block_founders,
                                    return_haplotype=return_haplotype,
                                    return_genotype_posterior=need_genotype_posterior,
                                )

                        t_after_hmm = time.perf_counter()
                        task_seconds = float(sum(result.seconds_hmm for result in task_results))
                        task_max_rss = float(max((result.rss_mb for result in task_results), default=0.0))
                        block_timing = self._finalize_and_write_block(
                            samples=samples,
                            sample_ids=sample_ids,
                            sample_ploidy=sample_ploidy,
                            generations=generations,
                            pedigree=pedigree,
                            block=block,
                            block_start_offset=block_start_offset,
                            evidence=evidence,
                            support_mask=support_mask,
                            artifacts=artifacts,
                            need_genotype_posterior=need_genotype_posterior,
                            requested_genotype_outputs=requested_genotype_outputs,
                            max_ploidy=max_ploidy,
                            block_t0=block_t0,
                            t_after_reads=t_after_reads,
                            t_after_hmm=t_after_hmm,
                            rss_block_start=rss_block_start,
                            rss_after_reads=rss_after_reads,
                            dask_extra={
                                "executor": "dask",
                                "dask_tasks": int(len(task_results)),
                                "dask_task_seconds_sum": task_seconds,
                                "dask_task_max_rss_mb": task_max_rss,
                                "dask_block_size": int(effective_block_size),
                                "dask_sample_batch_size": int(sample_batch_size),
                            },
                        )
                        timings.append(block_timing)
                    finally:
                        if read_evidence is not None:
                            read_evidence.release()
                        evidence.release()
                        if self.config.gc_collect_every_block:
                            gc.collect()
        finally:
            if int(read_samples.shape[0]) > 0:
                self.read_extractor.close()
            hold_seconds = max(float(self.config.dask_dashboard_hold_seconds), 0.0)
            if hold_seconds > 0.0 and dashboard_link:
                print(
                    f"Holding Dask dashboard at {dashboard_link} for {hold_seconds:.1f} seconds before shutdown.",
                    flush=True,
                )
                time.sleep(hold_seconds)
            client.close()
            cluster.close()

        if task_stream_path is not None and task_stream is not None:
            write_task_stream_artifact(task_stream_path, list(getattr(task_stream, "data", [])))

        timings_path = self.output_dir / "stage_timings.json"
        timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
        if self.config.profile_memory and timings:
            peak_rss = max(float(row.get("rss_mb_after_write", 0.0)) for row in timings)
            peak_hmm_rss = max(float(row.get("rss_mb_after_hmm", 0.0)) for row in timings)
            mem_summary = {
                "peak_rss_mb": peak_rss,
                "peak_hmm_rss_mb": peak_hmm_rss,
                "blocks_profiled": len(timings),
            }
            (self.output_dir / "memory_profile_summary.json").write_text(
                json.dumps(mem_summary, indent=2),
                encoding="utf-8",
            )
        dask_summary = {
            "executor": "dask",
            "scheduler": self.config.dask_scheduler,
            "dashboard_url": dashboard_link,
            "n_workers": int(n_workers),
            "threads_per_worker": int(self.config.dask_threads_per_worker),
            "processes": bool(self.config.dask_processes),
            "memory_limit": memory_limit,
            "chunk_plan": chunk_plan.to_dict(),
            "effective_block_size": int(effective_block_size),
            "sample_batch_size": int(sample_batch_size),
            "performance_report": (str(performance_report_path) if performance_report_path is not None else None),
            "task_stream": (str(task_stream_path) if task_stream_path is not None else None),
            "dashboard_hold_seconds": float(self.config.dask_dashboard_hold_seconds),
            "total_hmm_tasks": int(total_tasks),
            "mutable_founder_forced_full_ploidy_group_batches": bool(mutable_founder_forced_batches),
            "task_diagnostics": task_diagnostics,
        }
        (self.output_dir / "dask_run_summary.json").write_text(
            json.dumps(dask_summary, indent=2),
            encoding="utf-8",
        )

    def _write_hmm_outputs(
        self,
        sample_ids: np.ndarray,
        block,
        dosage: np.ndarray,
        artifacts,
        *,
        support_mask: np.ndarray | None = None,
        genotype_posterior: np.ndarray | None = None,
        genotype_call: np.ndarray | None = None,
    ) -> None:
        positions = block.dataframe["POS"].to_numpy(dtype=np.int64)
        n_samples, n_positions = dosage.shape
        n_rows = n_samples * n_positions
        sample_col = np.repeat(sample_ids, n_positions)
        position_col = np.tile(positions, n_samples)
        block_col = np.repeat(block.block_id, n_rows)
        dosage_table = pa.table(
            {
                "sample_id": sample_col,
                "chromosome": np.repeat(self.config.chromosome, n_rows),
                "position": position_col,
                "dosage": dosage.reshape(-1).astype(np.float32),
                "block_id": block_col,
            }
        )
        write_parquet(
            dosage_table,
            self.output_dir / "dosage" / f"block={block.block_id:06d}.parquet",
            self.config.compression,
            self.config.compression_level,
            row_group_size=262_144,
        )

        if self.io_config.write_support_mask:
            if support_mask is None:
                raise RuntimeError("Supporting-read mask missing while write_support_mask=True")
            support_table = pa.table(
                {
                    "sample_id": sample_col,
                    "chromosome": np.repeat(self.config.chromosome, n_rows),
                    "position": position_col,
                    "has_supporting_read": support_mask.reshape(-1).astype(np.bool_, copy=False),
                    "block_id": block_col,
                }
            )
            write_parquet(
                support_table,
                self.output_dir / "support_mask" / f"block={block.block_id:06d}.parquet",
                self.config.compression,
                self.config.compression_level,
                row_group_size=262_144,
            )

        rate_table = pa.table(
            {
                "chromosome": np.repeat(self.config.chromosome, positions.shape[0]),
                "position": positions,
                "recombination_rate": artifacts.recombination_rate.astype(np.float32),
                "block_id": np.repeat(block.block_id, positions.shape[0]),
            }
        )
        write_parquet(
            rate_table,
            self.output_dir / "recombination" / f"block={block.block_id:06d}.parquet",
            self.config.compression,
            self.config.compression_level,
        )

        if self.io_config.write_transitions:
            transition_rows = artifacts.switch_probability.size
            transitions = pa.table(
                {
                    "sample_id": sample_col,
                    "chromosome": np.repeat(self.config.chromosome, transition_rows),
                    "position": position_col,
                    "switch_probability": artifacts.switch_probability.reshape(-1).astype(np.float32),
                    "stay_probability": artifacts.stay_probability.reshape(-1).astype(np.float32),
                    "offdiag_probability": artifacts.offdiag_probability.reshape(-1).astype(np.float32),
                    "block_id": np.repeat(block.block_id, transition_rows),
                }
            )
            write_parquet(
                transitions,
                self.output_dir / "transitions" / f"block={block.block_id:06d}.parquet",
                self.config.compression,
                self.config.compression_level,
            )

        if self.io_config.transition_output == "full" and artifacts.transition_probability is not None:
            full = artifacts.transition_probability.reshape(
                artifacts.transition_probability.shape[0],
                artifacts.transition_probability.shape[1],
                -1,
            )
            values = pa.array(full.reshape(-1), type=pa.float32())
            vector = pa.FixedSizeListArray.from_arrays(values, full.shape[2])
            full_table = pa.table(
                {
                    "sample_id": sample_col,
                    "chromosome": np.repeat(self.config.chromosome, full.shape[0] * full.shape[1]),
                    "position": position_col,
                    "transition_probability": vector,
                    "block_id": np.repeat(block.block_id, full.shape[0] * full.shape[1]),
                }
            )
            write_parquet(
                full_table,
                self.output_dir / "transitions_full" / f"block={block.block_id:06d}.parquet",
                self.config.compression,
                self.config.compression_level,
            )

        if self.io_config.write_haplotype_probabilities:
            k = int(artifacts.founder_alt_prob.shape[0])
            if artifacts.haplotype_posterior is None:
                raise RuntimeError("HMM haplotype posterior missing while write_haplotype_probabilities=True")
            hap_probability = artifacts.haplotype_posterior.astype(np.float32, copy=False)
            if self._sample_ploidy is not None and self._sample_ploidy.shape[0] == hap_probability.shape[0]:
                ploidy_scale = self._sample_ploidy.astype(np.float32, copy=False)[:, None, None]
                hap_dosage = (hap_probability * ploidy_scale).astype(np.float32, copy=False)
            elif self.config.ploidy_mode == "diploid":
                hap_dosage = (2.0 * hap_probability).astype(np.float32, copy=False)
            else:
                hap_dosage = hap_probability
            hap_dosage = hap_dosage.astype(np.float32, copy=False)
            hap_probability = hap_probability.astype(np.float32, copy=False)
            hap_dosage_values = pa.array(hap_dosage.reshape(-1), type=pa.float32())
            hap_probability_values = pa.array(hap_probability.reshape(-1), type=pa.float32())
            hap_dosage_vector = pa.FixedSizeListArray.from_arrays(hap_dosage_values, k)
            hap_probability_vector = pa.FixedSizeListArray.from_arrays(hap_probability_values, k)
            hap_table = pa.table(
                {
                    "sample_id": sample_col,
                    "chromosome": np.repeat(self.config.chromosome, n_rows),
                    "position": position_col,
                    "hap_dosage": hap_dosage_vector,
                    "hap_probability": hap_probability_vector,
                    "block_id": block_col,
                }
            )
            write_parquet(
                hap_table,
                self.output_dir / "haplotype_probabilities" / f"block={block.block_id:06d}.parquet",
                self.config.compression,
                self.config.compression_level,
                row_group_size=262_144,
            )

        if self.io_config.write_genotype_posteriors:
            if genotype_posterior is None:
                raise RuntimeError("Genotype posterior missing while write_genotype_posteriors=True")
            gp_values = pa.array(
                genotype_posterior.astype(np.float32, copy=False).reshape(-1),
                type=pa.float32(),
            )
            gp_vector = pa.FixedSizeListArray.from_arrays(gp_values, int(genotype_posterior.shape[2]))
            gp_table = pa.table(
                {
                    "sample_id": sample_col,
                    "chromosome": np.repeat(self.config.chromosome, n_rows),
                    "position": position_col,
                    "genotype_posterior": gp_vector,
                    "block_id": block_col,
                }
            )
            write_parquet(
                gp_table,
                self.output_dir / "genotype_posteriors" / f"block={block.block_id:06d}.parquet",
                self.config.compression,
                self.config.compression_level,
                row_group_size=262_144,
            )

        if self.io_config.write_genotype_calls:
            if genotype_call is None:
                raise RuntimeError("Genotype call missing while write_genotype_calls=True")
            gt_table = pa.table(
                {
                    "sample_id": sample_col,
                    "chromosome": np.repeat(self.config.chromosome, n_rows),
                    "position": position_col,
                    "genotype_call": genotype_call.reshape(-1).astype(np.int8, copy=False),
                    "block_id": block_col,
                }
            )
            write_parquet(
                gt_table,
                self.output_dir / "genotype_calls" / f"block={block.block_id:06d}.parquet",
                self.config.compression,
                self.config.compression_level,
                row_group_size=262_144,
            )

        founder_idx, pos_idx = np.indices(artifacts.founder_alt_prob.shape)
        founder_table = pa.table(
            {
                "chromosome": np.repeat(self.config.chromosome, artifacts.founder_alt_prob.size),
                "position": positions[pos_idx.reshape(-1)],
                "founder": founder_idx.reshape(-1),
                "alt_prob": artifacts.founder_alt_prob.reshape(-1).astype(np.float32),
                "block_id": np.repeat(block.block_id, artifacts.founder_alt_prob.size),
            }
        )
        write_parquet(
            founder_table,
            self.output_dir / "founder_updates" / f"block={block.block_id:06d}.parquet",
            self.config.compression,
            self.config.compression_level,
        )
