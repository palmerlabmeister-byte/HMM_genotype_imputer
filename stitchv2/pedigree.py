from __future__ import annotations

import numpy as np
from scipy import sparse


def smooth_dosage_with_pedigree(
    dosage: np.ndarray,
    pedigree: sparse.csr_matrix | None,
    strength: float,
) -> np.ndarray:
    if pedigree is None or strength <= 0.0:
        return dosage
    parent_degree = np.asarray(pedigree.sum(axis=1)).reshape(-1, 1)
    parent_degree[parent_degree == 0] = 1.0
    parent_mean = pedigree @ dosage / parent_degree
    return (1.0 - strength) * dosage + strength * parent_mean
