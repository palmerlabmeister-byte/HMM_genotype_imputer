from __future__ import annotations

import numpy as np
import pandas as pd

from stitchcont.pedigree import pedigree_table_to_graph, smooth_dosage_with_pedigree, apply_transmission_message_passing


def test_smooth_dosage_leaves_samples_without_parents_unchanged():
    samples = np.asarray(["father", "mother", "child"], dtype=object)
    ped = pd.DataFrame({"sample_id": ["child"], "father_id": ["father"], "mother_id": ["mother"]})
    graph = pedigree_table_to_graph(ped, samples)
    dosage = np.asarray([[0.25, 0.50], [1.50, 1.75], [0.0, 2.0]], dtype=np.float32)
    smoothed = smooth_dosage_with_pedigree(dosage, graph, strength=1.0)
    np.testing.assert_allclose(smoothed[0], dosage[0])
    np.testing.assert_allclose(smoothed[1], dosage[1])
    np.testing.assert_allclose(smoothed[2], 0.5 * (dosage[0] + dosage[1]))


def test_transmission_message_pushes_unobserved_child_toward_mendelian_het():
    samples = np.asarray(["father", "mother", "child"], dtype=object)
    ped = pd.DataFrame({"sample_id": ["child"], "father_id": ["father"], "mother_id": ["mother"]})
    graph = pedigree_table_to_graph(ped, samples)
    gp = np.zeros((3, 3, 3), dtype=np.float32)
    gp[0, :, 0] = 1.0
    gp[1, :, 2] = 1.0
    gp[2, :, :] = 1.0 / 3.0
    support = np.asarray([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=bool)
    res = apply_transmission_message_passing(
        gp,
        graph,
        positions=np.arange(3, dtype=np.int64),
        generations=np.ones(3, dtype=np.float32),
        support_mask=support,
        strength=0.8,
        iterations=3,
    )
    assert np.all(res.genotype_posterior[2, :, 1] > 0.65)
    np.testing.assert_allclose(res.dosage[2], np.ones(3, dtype=np.float32), atol=1e-5)
