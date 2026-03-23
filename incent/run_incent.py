"""Convenience wrapper matching the legacy five-value INCENT output.

This module keeps the legacy five-value output unchanged while routing
through the v2 CAST stack so the wrapper exercises the broader LRF,
robust SE(2), adaptive partial OT, cVAE, and LDDMM components.
"""

from __future__ import annotations

import os

import numpy as np

from .core import _preprocess, _to_np
from .cast_v2 import pairwise_align_cast_v2
from .pose import apply_pose, estimate_pose


def run_incent(
    sliceA,
    sliceB,
    data1,
    data2,
    method_name,
    alpha: float = 0.8,
    beta: float = 0.9,
    gamma: float = 0.5,
    radius: float = 100,
    numItermax: int = 20000,
    use_gpu: bool = True,
    overwrite: bool = True,
    neighborhood_dissimilarity: str = "jsd",
):
    """Run the INCENT-SE spatiotemporal aligner and return legacy diagnostics.

    The returned tuple is:
        pi12, init_nb, init_gene, final_nb, final_gene
    where the initial and final values are the mean costs under a uniform
    coupling and under the learned coupling respectively.
    """

    file_path = os.path.join(os.getcwd(), "local_data", str(method_name))
    os.makedirs(file_path, exist_ok=True)

    # 1) Apply pose up front so the aligner and the bookkeeping use the same
    # coordinates.
    theta, tx, ty, _ = estimate_pose(sliceA, sliceB, grid_size=256, verbose=True)
    sliceA_pose = apply_pose(sliceA, theta, tx, ty, inplace=False)

    # 2) Run the v2 CAST pipeline on the pose-normalised coordinates.
    # This activates the v2 descriptor, robust SE(2), adaptive OT, cVAE,
    # and optional LDDMM code paths inside cast_v2.py and its helpers.
    result = pairwise_align_cast_v2(
        sliceA=sliceA_pose,
        sliceB=sliceB,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        radius=radius,
        filePath=file_path,
        use_lrf=True,
        use_magsac=True,
        do_lo_ransac=True,
        use_adaptive_ot=True,
        cross_timepoint=True,
        use_lddmm=True,
        sliceA_name=data1,
        sliceB_name=data2,
        use_gpu=use_gpu,
        gpu_verbose=False,
        verbose=True,
        numItermax=numItermax,
        overwrite=overwrite,
        neighborhood_dissimilarity=neighborhood_dissimilarity,
        return_diagnostics=True,
    )

    pi12 = result[0]

    # 3) Recompute the matrices used for the old-style diagnostics.
    with open(os.devnull, "w") as log_file:
        p = _preprocess(
            sliceA_pose,
            sliceB,
            alpha,
            beta,
            gamma,
            radius,
            file_path,
            use_rep=None,
            G_init=None,
            a_distribution=None,
            b_distribution=None,
            numItermax=numItermax,
            backend=None,
            use_gpu=use_gpu,
            gpu_verbose=False,
            sliceA_name=data1,
            sliceB_name=data2,
            overwrite=overwrite,
            neighborhood_dissimilarity=neighborhood_dissimilarity,
            logFile=log_file,
        )

    m_neighbor = _to_np(p["M2"])
    m_gene_cos = _to_np(p["cosine_dist_gene_expr"])
    n_a, n_b = p["sliceA"].n_obs, p["sliceB"].n_obs
    g0 = np.full((n_a, n_b), 1.0 / (n_a * n_b), dtype=np.float64)

    init_nb = float(np.sum(m_neighbor * g0))
    init_gene = float(np.sum(m_gene_cos * g0))
    final_nb = float(np.sum(m_neighbor * pi12))
    final_gene = float(np.sum(m_gene_cos * pi12))

    return pi12, init_nb, init_gene, final_nb, final_gene