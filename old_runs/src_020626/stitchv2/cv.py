from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .calibration import (
    calibrate_genotype_posterior,
    predict_lightgbm_posterior,
    read_log_likelihood_from_posterior,
    train_lightgbm_multiclass_calibrator,
)
from .config import HMMConfig
from .founders import FounderPanel
from .hmm import JAXStitchHMM
from .io import iter_position_blocks, validate_samples
from .pileup import PysamReadExtractor


def _macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    f1_vals: list[float] = []
    for cls in (0, 1, 2):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        denom = (2.0 * tp + fp + fn)
        f1_vals.append((2.0 * tp / denom) if denom > 0.0 else 0.0)
    return float(np.mean(f1_vals))


def _normalize_pseudo_truth(pseudo_truth: pd.DataFrame | None) -> pd.DataFrame | None:
    if pseudo_truth is None:
        return None
    df = pseudo_truth.copy()
    if "sample_id" not in df.columns or "position" not in df.columns:
        raise ValueError("pseudo_truth must contain columns: sample_id, position, and genotype or dosage_truth.")
    df["sample_id"] = df["sample_id"].astype(str)
    df["position"] = df["position"].astype(np.int64)
    if "genotype" in df.columns:
        df["genotype_truth"] = df["genotype"].astype(np.int8)
    elif "genotype_truth" in df.columns:
        df["genotype_truth"] = df["genotype_truth"].astype(np.int8)
    elif "dosage_truth" in df.columns:
        df["genotype_truth"] = np.clip(np.rint(df["dosage_truth"].astype(np.float32)), 0, 2).astype(np.int8)
    elif "dosage" in df.columns:
        df["genotype_truth"] = np.clip(np.rint(df["dosage"].astype(np.float32)), 0, 2).astype(np.int8)
    else:
        raise ValueError("pseudo_truth must include genotype/genotype_truth or dosage_truth/dosage.")
    if "dosage_truth" not in df.columns:
        df["dosage_truth"] = df["genotype_truth"].astype(np.float32)
    else:
        df["dosage_truth"] = df["dosage_truth"].astype(np.float32)
    return df[["sample_id", "position", "genotype_truth", "dosage_truth"]].drop_duplicates(
        subset=["sample_id", "position"],
        keep="first",
    )


def _make_folds(positions: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    if n_folds <= 1:
        return [positions.copy()]
    rng = np.random.default_rng(int(seed))
    shuffled = positions.copy()
    rng.shuffle(shuffled)
    splits = np.array_split(shuffled, int(n_folds))
    out = [np.sort(split.astype(np.int64, copy=False)) for split in splits if split.size > 0]
    return out


def _build_founder_panel_for_k(
    *,
    k: int,
    chromosome: str,
    positions_df: pd.DataFrame,
    base_founder_panel: FounderPanel | None,
) -> FounderPanel:
    n_pos = int(len(positions_df))
    positions = positions_df["POS"].to_numpy(dtype=np.int64)
    ref = positions_df["REF"].astype(str).to_numpy()
    alt = positions_df["ALT"].astype(str).to_numpy()
    if base_founder_panel is None:
        return FounderPanel(
            chromosome=chromosome,
            positions=positions,
            ref=ref,
            alt=alt,
            alt_prob=np.full((k, n_pos), 0.5, dtype=np.float32),
            immutable_mask=np.zeros(k, dtype=bool),
        )
    if base_founder_panel.n_positions != n_pos:
        raise ValueError("base_founder_panel and positions_df disagree on number of positions.")
    if base_founder_panel.n_founders >= k:
        return FounderPanel(
            chromosome=chromosome,
            positions=positions,
            ref=ref,
            alt=alt,
            alt_prob=base_founder_panel.alt_prob[:k].astype(np.float32, copy=True),
            immutable_mask=base_founder_panel.immutable_mask[:k].astype(bool, copy=True),
        )
    pad = k - base_founder_panel.n_founders
    alt_prob = np.concatenate(
        [
            base_founder_panel.alt_prob.astype(np.float32, copy=True),
            np.full((pad, n_pos), 0.5, dtype=np.float32),
        ],
        axis=0,
    )
    immutable_mask = np.concatenate(
        [base_founder_panel.immutable_mask.astype(bool, copy=True), np.zeros(pad, dtype=bool)],
        axis=0,
    )
    return FounderPanel(
        chromosome=chromosome,
        positions=positions,
        ref=ref,
        alt=alt,
        alt_prob=alt_prob,
        immutable_mask=immutable_mask,
    )


def _empty_fold_arrays(n_samples: int, n_holdout: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gp_sum = np.zeros((n_samples, n_holdout, 3), dtype=np.float32)
    ref_hold = np.zeros((n_samples, n_holdout), dtype=np.float32)
    alt_hold = np.zeros((n_samples, n_holdout), dtype=np.float32)
    oth_hold = np.zeros((n_samples, n_holdout), dtype=np.float32)
    return gp_sum, ref_hold, alt_hold, oth_hold


def run_realdata_cv_harness(
    *,
    samples: pd.DataFrame,
    positions_df: pd.DataFrame,
    chromosome: str,
    output_dir: str | Path,
    base_founder_panel: FounderPanel | None = None,
    pseudo_truth: pd.DataFrame | None = None,
    k_values: Sequence[int] = (8,),
    ngen_values: Sequence[float] = (1.0,),
    s_values: Sequence[int] = (2,),
    seeds: Sequence[int] = (0, 1, 2),
    n_folds: int = 5,
    holdout_fraction: float = 0.2,
    block_size: int = 1000,
    hmm_backend: str = "jax",
    read_mode: str = "read_stream",
    read_stream_backend: str = "auto",
    io_workers: int = 1,
    htslib_threads_per_file: int = 1,
    use_fragment_likelihood: bool = True,
    fragment_likelihood_mode: str = "replace",
    fragment_coupling_model: str = "stitch_parity",
    fragment_max_difference_between_reads: float = 100.0,
    fragment_max_emission_matrix_difference: float = 1000.0,
    fragment_rescale_read_likelihood: bool = True,
    genotype_posterior_temperature: float = 0.35,
    genotype_posterior_blend: float = 0.35,
    founder_init_jitter: float = 0.0,
    memory_map_read_matrices: bool = False,
    memory_map_dir: str | Path | None = None,
    lightgbm_post_calibrator: bool = False,
) -> dict[str, object]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = validate_samples(samples)
    positions_df = positions_df.copy().sort_values("POS").reset_index(drop=True)
    positions = positions_df["POS"].to_numpy(dtype=np.int64)
    sample_ids = samples["sample_id"].astype(str).to_numpy()
    n_samples = int(sample_ids.size)
    if n_samples == 0:
        raise ValueError("No samples provided.")
    if len(positions) == 0:
        raise ValueError("No positions provided.")
    if not (0.0 < float(holdout_fraction) < 1.0):
        raise ValueError("holdout_fraction must be in (0,1).")

    pseudo_truth_df = _normalize_pseudo_truth(pseudo_truth)
    folds = _make_folds(positions, n_folds=n_folds, seed=int(seeds[0]) if len(seeds) > 0 else 0)

    seed_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []
    calibration_rows: list[pd.DataFrame] = []
    combo_id = 0
    generation_base = samples["generation"].to_numpy(dtype=np.float32, copy=False)

    for k in k_values:
        founder_panel = _build_founder_panel_for_k(
            k=int(k),
            chromosome=chromosome,
            positions_df=positions_df,
            base_founder_panel=base_founder_panel,
        )
        for ngen_scale in ngen_values:
            generations = generation_base * float(ngen_scale)
            for s_iter in s_values:
                combo_id += 1
                combo_name = f"K={int(k)}_nGen={float(ngen_scale):g}_S={int(s_iter)}"
                for fold_id, holdout_pos in enumerate(folds):
                    if holdout_pos.size == 0:
                        continue
                    hold_map = {int(pos): idx for idx, pos in enumerate(holdout_pos.tolist())}
                    n_hold = int(holdout_pos.size)
                    gp_sum, ref_hold, alt_hold, oth_hold = _empty_fold_arrays(n_samples, n_hold)
                    depth_hold = np.zeros((n_samples, n_hold), dtype=np.float32)
                    alt_frac_hold = np.zeros((n_samples, n_hold), dtype=np.float32)

                    for seed in seeds:
                        gp_seed = np.zeros((n_samples, n_hold, 3), dtype=np.float32)
                        hmm = JAXStitchHMM(
                            HMMConfig(
                                n_founders=int(k),
                                ploidy_mode="diploid",
                                em_iterations=int(s_iter),
                                backend=hmm_backend,
                                use_quality_weights=True,
                                use_fragment_likelihood=bool(use_fragment_likelihood),
                                fragment_likelihood_mode=fragment_likelihood_mode,
                                fragment_coupling_model=fragment_coupling_model,  # type: ignore[arg-type]
                                fragment_max_difference_between_reads=float(fragment_max_difference_between_reads),
                                fragment_max_emission_matrix_difference=float(fragment_max_emission_matrix_difference),
                                fragment_rescale_read_likelihood=bool(fragment_rescale_read_likelihood),
                                random_seed=int(seed),
                                founder_init_jitter=float(founder_init_jitter),
                            )
                        )
                        extractor = PysamReadExtractor(
                            chromosome,
                            mode=read_mode,  # type: ignore[arg-type]
                            read_stream_backend=read_stream_backend,  # type: ignore[arg-type]
                            merge_fragments_by_query=True,
                            io_workers=int(io_workers),
                            htslib_threads_per_file=int(htslib_threads_per_file),
                            memory_map_read_matrices=bool(memory_map_read_matrices),
                            memory_map_dir=memory_map_dir,
                        )
                        extractor.open(samples)
                        try:
                            for block in iter_position_blocks(positions_df, int(block_size)):
                                evidence = extractor.extract_block(samples, block)
                                try:
                                    block_pos = block.dataframe["POS"].to_numpy(dtype=np.int64)
                                    block_hold_mask = np.isin(block_pos, holdout_pos, assume_unique=False)
                                    hold_local_idx = np.flatnonzero(block_hold_mask)
                                    if hold_local_idx.size > 0:
                                        hold_global_idx = np.asarray(
                                            [hold_map[int(block_pos[i])] for i in hold_local_idx.tolist()],
                                            dtype=np.int64,
                                        )
                                        if int(seed) == int(seeds[0]):
                                            ref_hold[:, hold_global_idx] = evidence.ref_count[:, hold_local_idx].astype(np.float32, copy=False)
                                            alt_hold[:, hold_global_idx] = evidence.alt_count[:, hold_local_idx].astype(np.float32, copy=False)
                                            oth_hold[:, hold_global_idx] = evidence.other_count[:, hold_local_idx].astype(np.float32, copy=False)
                                            depth_block = evidence.depth[:, hold_local_idx].astype(np.float32, copy=False)
                                            depth_hold[:, hold_global_idx] = depth_block
                                            alt_frac_hold[:, hold_global_idx] = (
                                                alt_hold[:, hold_global_idx] + 0.5
                                            ) / np.clip(
                                                ref_hold[:, hold_global_idx]
                                                + alt_hold[:, hold_global_idx]
                                                + oth_hold[:, hold_global_idx]
                                                + 1.0,
                                                1.0,
                                                None,
                                            )
                                        # true holdout: remove read evidence from held-out positions.
                                        evidence.ref_count[:, hold_local_idx] = 0
                                        evidence.alt_count[:, hold_local_idx] = 0
                                        evidence.other_count[:, hold_local_idx] = 0
                                        evidence.ref_weight[:, hold_local_idx] = 0.0
                                        evidence.alt_weight[:, hold_local_idx] = 0.0
                                        evidence.other_weight[:, hold_local_idx] = 0.0

                                    block_founders = founder_panel.slice(
                                        block.block_id * int(block_size),
                                        min((block.block_id + 1) * int(block_size), founder_panel.n_positions),
                                    )
                                    artifacts = hmm.run(
                                        founder_panel=block_founders,
                                        ref_count=evidence.ref_count,
                                        alt_count=evidence.alt_count,
                                        other_count=evidence.other_count,
                                        ref_weight=evidence.ref_weight,
                                        alt_weight=evidence.alt_weight,
                                        other_weight=evidence.other_weight,
                                        generations=generations,
                                        return_genotype_posterior=True,
                                        return_haplotype_posterior=False,
                                        fragment_sample_offsets=evidence.fragment_sample_offsets,
                                        fragment_center_idx=evidence.fragment_center_idx,
                                        fragment_obs_offsets=evidence.fragment_obs_offsets,
                                        fragment_obs_pos_idx=evidence.fragment_obs_pos_idx,
                                        fragment_obs_code=evidence.fragment_obs_code,
                                        fragment_obs_qual=evidence.fragment_obs_qual,
                                    )
                                    gp_cal = calibrate_genotype_posterior(
                                        artifacts.genotype_posterior,
                                        dosage=artifacts.dosage,
                                        depth=evidence.depth.astype(np.float32, copy=False),
                                        temperature=genotype_posterior_temperature,
                                        blend=genotype_posterior_blend,
                                    )
                                    if hold_local_idx.size > 0:
                                        gp_hold = gp_cal[:, hold_local_idx, :].astype(np.float32, copy=False)
                                        gp_seed[:, hold_global_idx, :] = gp_hold
                                finally:
                                    evidence.release()
                        finally:
                            extractor.close()

                        gp_sum += gp_seed
                        _, ll_seed = read_log_likelihood_from_posterior(
                            ref_count=ref_hold,
                            alt_count=alt_hold,
                            other_count=oth_hold,
                            posterior=gp_seed,
                        )
                        ds_seed = gp_seed[:, :, 1] + 2.0 * gp_seed[:, :, 2]
                        gt_seed = np.argmax(gp_seed, axis=2).astype(np.int8, copy=False)
                        seed_row: dict[str, object] = {
                            "combo_id": combo_id,
                            "combo_name": combo_name,
                            "K": int(k),
                            "nGen_scale": float(ngen_scale),
                            "S": int(s_iter),
                            "seed": int(seed),
                            "fold": int(fold_id),
                            "holdout_n_positions": int(n_hold),
                            "heldout_log_likelihood": float(ll_seed),
                            "heldout_log_likelihood_per_obs": float(
                                ll_seed / max(float(np.sum(ref_hold + alt_hold + oth_hold)), 1.0)
                            ),
                        }
                        if pseudo_truth_df is not None:
                            seed_pred_df = pd.DataFrame(
                                {
                                    "sample_id": np.repeat(sample_ids, n_hold),
                                    "position": np.tile(holdout_pos, n_samples),
                                    "dosage_pred": ds_seed.reshape(-1),
                                    "genotype_pred": gt_seed.reshape(-1),
                                }
                            )
                            merged = seed_pred_df.merge(pseudo_truth_df, on=["sample_id", "position"], how="inner")
                            if not merged.empty:
                                y_true = merged["genotype_truth"].to_numpy(dtype=np.int8, copy=False)
                                y_pred = merged["genotype_pred"].to_numpy(dtype=np.int8, copy=False)
                                d_true = merged["dosage_truth"].to_numpy(dtype=np.float32, copy=False)
                                d_pred = merged["dosage_pred"].to_numpy(dtype=np.float32, copy=False)
                                seed_row.update(
                                    {
                                        "n_truth_calls": int(len(merged)),
                                        "accuracy": float(np.mean(y_true == y_pred)),
                                        "macro_f1": _macro_f1_score(y_true, y_pred),
                                        "rmse": float(np.sqrt(np.mean((d_true - d_pred) ** 2))),
                                    }
                                )
                            else:
                                seed_row.update({"n_truth_calls": 0, "accuracy": float("nan"), "macro_f1": float("nan"), "rmse": float("nan")})
                        seed_rows.append(seed_row)

                    gp_ens = (gp_sum / max(float(len(seeds)), 1.0)).astype(np.float32, copy=False)
                    _, ll_ens = read_log_likelihood_from_posterior(
                        ref_count=ref_hold,
                        alt_count=alt_hold,
                        other_count=oth_hold,
                        posterior=gp_ens,
                    )
                    ds_ens = gp_ens[:, :, 1] + 2.0 * gp_ens[:, :, 2]
                    gt_ens = np.argmax(gp_ens, axis=2).astype(np.int8, copy=False)
                    ens_row: dict[str, object] = {
                        "combo_id": combo_id,
                        "combo_name": combo_name,
                        "K": int(k),
                        "nGen_scale": float(ngen_scale),
                        "S": int(s_iter),
                        "fold": int(fold_id),
                        "holdout_n_positions": int(n_hold),
                        "heldout_log_likelihood": float(ll_ens),
                        "heldout_log_likelihood_per_obs": float(
                            ll_ens / max(float(np.sum(ref_hold + alt_hold + oth_hold)), 1.0)
                        ),
                    }
                    if pseudo_truth_df is not None:
                        ens_pred_df = pd.DataFrame(
                            {
                                "sample_id": np.repeat(sample_ids, n_hold),
                                "position": np.tile(holdout_pos, n_samples),
                                "dosage_pred": ds_ens.reshape(-1),
                                "genotype_pred": gt_ens.reshape(-1),
                                "gp0": gp_ens[:, :, 0].reshape(-1),
                                "gp1": gp_ens[:, :, 1].reshape(-1),
                                "gp2": gp_ens[:, :, 2].reshape(-1),
                                "depth": depth_hold.reshape(-1),
                                "alt_fraction": alt_frac_hold.reshape(-1),
                                "ref_count": ref_hold.reshape(-1),
                                "alt_count": alt_hold.reshape(-1),
                                "other_count": oth_hold.reshape(-1),
                                "fold": np.repeat(fold_id, n_samples * n_hold),
                                "combo_id": np.repeat(combo_id, n_samples * n_hold),
                            }
                        )
                        merged = ens_pred_df.merge(pseudo_truth_df, on=["sample_id", "position"], how="inner")
                        if not merged.empty:
                            y_true = merged["genotype_truth"].to_numpy(dtype=np.int8, copy=False)
                            y_pred = merged["genotype_pred"].to_numpy(dtype=np.int8, copy=False)
                            d_true = merged["dosage_truth"].to_numpy(dtype=np.float32, copy=False)
                            d_pred = merged["dosage_pred"].to_numpy(dtype=np.float32, copy=False)
                            ens_row.update(
                                {
                                    "n_truth_calls": int(len(merged)),
                                    "accuracy": float(np.mean(y_true == y_pred)),
                                    "macro_f1": _macro_f1_score(y_true, y_pred),
                                    "rmse": float(np.sqrt(np.mean((d_true - d_pred) ** 2))),
                                }
                            )
                            if lightgbm_post_calibrator:
                                calibration_rows.append(
                                    merged[
                                        [
                                            "combo_id",
                                            "fold",
                                            "dosage_pred",
                                            "gp0",
                                            "gp1",
                                            "gp2",
                                            "depth",
                                            "alt_fraction",
                                            "ref_count",
                                            "alt_count",
                                            "other_count",
                                            "genotype_truth",
                                        ]
                                    ].copy()
                                )
                        else:
                            ens_row.update({"n_truth_calls": 0, "accuracy": float("nan"), "macro_f1": float("nan"), "rmse": float("nan")})
                    ensemble_rows.append(ens_row)

    seed_df = pd.DataFrame(seed_rows)
    ensemble_df = pd.DataFrame(ensemble_rows)
    seed_df.to_parquet(out_dir / "cv_seed_metrics.parquet", index=False)
    ensemble_df.to_parquet(out_dir / "cv_ensemble_metrics.parquet", index=False)
    combo_summary = (
        ensemble_df.groupby(["combo_id", "combo_name", "K", "nGen_scale", "S"], as_index=False)
        .agg(
            mean_ll=("heldout_log_likelihood", "mean"),
            mean_ll_per_obs=("heldout_log_likelihood_per_obs", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_rmse=("rmse", "mean"),
        )
        .sort_values(["mean_ll_per_obs", "mean_accuracy"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )
    combo_summary.to_parquet(out_dir / "cv_combo_summary.parquet", index=False)

    lightgbm_df = pd.DataFrame()
    if lightgbm_post_calibrator and calibration_rows:
        cal_df = pd.concat(calibration_rows, ignore_index=True)
        lgb_rows: list[dict[str, object]] = []
        for combo_value, combo_block in cal_df.groupby("combo_id"):
            folds_here = sorted(combo_block["fold"].unique().tolist())
            for fold in folds_here:
                train_df = combo_block.loc[combo_block["fold"] != fold]
                test_df = combo_block.loc[combo_block["fold"] == fold]
                if train_df.empty or test_df.empty:
                    continue
                feature_cols = ["dosage_pred", "gp0", "gp1", "gp2", "depth", "alt_fraction"]
                y_train = train_df["genotype_truth"].to_numpy(dtype=np.int32, copy=False)
                if np.unique(y_train).size < 2:
                    continue
                model = train_lightgbm_multiclass_calibrator(
                    train_df[feature_cols].to_numpy(dtype=np.float32, copy=False),
                    y_train,
                    seed=0,
                )
                pred_gp = predict_lightgbm_posterior(
                    model,
                    test_df[feature_cols].to_numpy(dtype=np.float32, copy=False),
                )
                pred_gt = np.argmax(pred_gp, axis=1).astype(np.int8, copy=False)
                y_true = test_df["genotype_truth"].to_numpy(dtype=np.int8, copy=False)
                ref = test_df["ref_count"].to_numpy(dtype=np.float32, copy=False)
                alt = test_df["alt_count"].to_numpy(dtype=np.float32, copy=False)
                oth = test_df["other_count"].to_numpy(dtype=np.float32, copy=False)
                ll_row, ll_total = read_log_likelihood_from_posterior(
                    ref_count=ref[:, None],
                    alt_count=alt[:, None],
                    other_count=oth[:, None],
                    posterior=pred_gp[:, None, :],
                )
                _ = ll_row
                lgb_rows.append(
                    {
                        "combo_id": int(combo_value),
                        "fold": int(fold),
                        "n_truth_calls": int(test_df.shape[0]),
                        "accuracy": float(np.mean(y_true == pred_gt)),
                        "macro_f1": _macro_f1_score(y_true, pred_gt),
                        "heldout_log_likelihood": float(ll_total),
                        "heldout_log_likelihood_per_obs": float(
                            ll_total / max(float(np.sum(ref + alt + oth)), 1.0)
                        ),
                    }
                )
        lightgbm_df = pd.DataFrame(lgb_rows)
        if not lightgbm_df.empty:
            lightgbm_df.to_parquet(out_dir / "cv_lightgbm_metrics.parquet", index=False)

    best_row = combo_summary.iloc[0].to_dict() if not combo_summary.empty else {}
    summary = {
        "output_dir": str(out_dir),
        "n_samples": int(n_samples),
        "n_positions": int(len(positions)),
        "n_folds": int(len(folds)),
        "holdout_fraction_requested": float(holdout_fraction),
        "k_values": [int(x) for x in k_values],
        "ngen_values": [float(x) for x in ngen_values],
        "s_values": [int(x) for x in s_values],
        "seeds": [int(x) for x in seeds],
        "best_combo": best_row,
        "lightgbm_rows": int(lightgbm_df.shape[0]),
    }
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
