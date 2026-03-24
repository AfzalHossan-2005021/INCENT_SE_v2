"""
bispa.py -- Bidirectional Symmetric Partial Alignment (BISPA) for INCENT-SE

BISPA supersedes RAPA by treating both slices symmetrically.
See docstrings on individual functions for details.
"""

import os
import time
import datetime
import warnings
import numpy as np
from typing import Optional, Tuple, List
from anndata import AnnData
from ._gpu import resolve_device, to_torch, to_numpy


# =========================================================================
# Stage 1: Symmetric spatial decomposition (same function for both slices)
# =========================================================================

def decompose_slice(
    adata,
    n_neighbors=15,
    resolution=None,
    min_community_size_frac=0.15,
    target_min_region_frac=0.20,
    slice_label="slice",
    verbose=True,
):
    """
    Unsupervised spatial decomposition of one slice into K communities.

    Works identically for sliceA and sliceB -- no asymmetric assumptions.

    The adaptive Leiden binary search finds the largest resolution where
    every community covers >= target_min_region_frac of n_cells:
      brain (2 hemispheres)  -> target_min_region_frac=0.20 -> K=2
      heart (4 chambers)     -> target_min_region_frac=0.10 -> K<=4
      liver (5 lobes)        -> target_min_region_frac=0.08 -> K<=5
      single region (K=1)    -> any target works

    Parameters
    ----------
    adata : AnnData with .obsm['spatial']
    n_neighbors : int -- kNN graph connectivity
    resolution : float or None -- None = adaptive (recommended)
    min_community_size_frac : float -- merge communities smaller than this
    target_min_region_frac : float -- adaptive search: every community >= this
    slice_label : str -- label for verbose messages
    verbose : bool

    Returns
    -------
    labels : (n_cells,) int32 -- community index 0..K-1 for each cell
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        warnings.warn(
            "[BISPA] leidenalg/igraph not found -- spectral clustering fallback.\n"
            "Install: pip install leidenalg igraph", stacklevel=2)
        return _spectral_fallback(adata, verbose=verbose)

    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm["spatial"].astype(np.float64)
    n = len(coords)

    if verbose:
        print(f"[BISPA decompose:{slice_label}] kNN graph (k={n_neighbors}) on {n} cells ...")

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree")
    nn.fit(coords)
    dists, indices = nn.kneighbors(coords)
    dists, indices = dists[:, 1:], indices[:, 1:]

    sigma = np.median(dists)
    weights = np.exp(-dists ** 2 / (2 * sigma ** 2)).ravel()
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices.ravel()

    G_ig = ig.Graph(n=n, edges=list(zip(rows.tolist(), cols.tolist())), directed=False)
    G_ig.es["weight"] = weights.tolist()

    if resolution is not None:
        res_final = resolution
        if verbose:
            print(f"[BISPA decompose:{slice_label}] fixed resolution={resolution}")
    else:
        # Binary search: find LARGEST resolution where every community
        # covers >= target_min_region_frac of n (i.e. coarsest valid partition).
        # lo=very low (K=1, always passes); hi=1.0 (K=many, most fail).
        lo, hi = 1e-6, 1.0
        best_res = lo
        best_lbl = None

        if verbose:
            print(f"[BISPA decompose:{slice_label}] adaptive search "
                  f"(target_min_frac={target_min_region_frac:.2f}) ...")

        for _ in range(25):
            mid = (lo + hi) / 2.0
            part = leidenalg.find_partition(
                G_ig, leidenalg.RBConfigurationVertexPartition,
                weights="weight", resolution_parameter=mid, seed=42)
            lbl = np.array(part.membership, dtype=np.int32)
            sizes = np.array([(lbl == k).sum() for k in np.unique(lbl)])
            min_f = sizes.min() / n

            if min_f >= target_min_region_frac:
                best_res, best_lbl = mid, lbl
                lo = mid   # accept: try finer (higher resolution)
            else:
                hi = mid   # reject: try coarser (lower resolution)

            if (hi - lo) < 1e-7:
                break

        if best_lbl is None:
            if verbose:
                print(f"[BISPA decompose:{slice_label}] no valid resolution -> K=1")
            return np.zeros(n, dtype=np.int32)

        res_final = best_res
        if verbose:
            K_found = len(np.unique(best_lbl))
            print(f"[BISPA decompose:{slice_label}] resolution={res_final:.6f} -> K={K_found}")

    part = leidenalg.find_partition(
        G_ig, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=res_final, seed=42)
    labels = np.array(part.membership, dtype=np.int32)
    labels = _merge_small(labels, coords, min_community_size_frac)

    # ── K=1 fallback: expression-guided spectral clustering ──────────────────
    # When Leiden finds K=1 (the spatial kNN graph is too well-connected to
    # split, e.g. brain hemispheres joined through corpus callosum), fall back
    # to spectral clustering on a COMBINED spatial+expression affinity.
    #
    # Why combined affinity works when pure-spatial Leiden fails:
    #   - The corpus callosum has a distinct cell-type composition (oligodendrocytes)
    #   - Cells in the left hemisphere are more similar to other left cells
    #     than to right cells in expression space
    #   - The combined affinity has a lower similarity across the midline,
    #     so spectral clustering correctly separates the two halves
    if len(np.unique(labels)) == 1:
        if verbose:
            print(f"[BISPA decompose:{slice_label}] K=1 from Leiden; "
                  f"trying expression-guided spectral clustering ...")
        labels = _expression_guided_spectral(adata, n_clusters=2, verbose=verbose)
        if verbose:
            K2 = len(np.unique(labels))
            for k in np.unique(labels):
                sz2 = (labels == k).sum()
                print(f"  [{slice_label}] C_{k}: {sz2} cells ({sz2/n*100:.1f}%)")
            print(f"[BISPA decompose:{slice_label}] Spectral K={K2} communities.")

    if verbose:
        K = len(np.unique(labels))
        for k in np.unique(labels):
            sz = (labels == k).sum()
            print(f"  [{slice_label}] C_{k}: {sz} cells ({sz/n*100:.1f}%)")
        print(f"[BISPA decompose:{slice_label}] K={K} communities.")

    return labels


def _expression_guided_spectral(adata, n_clusters=2, n_neighbors=15,
                                verbose=True):
    """
    Expression-guided spectral clustering for organs with spatially adjacent
    symmetric regions (e.g. brain hemispheres connected through corpus callosum).

    Pure spatial clustering fails because the corpus callosum spatially bridges
    both hemispheres.  This function builds a combined affinity matrix:
      W[i,j] = spatial_affinity[i,j] * expression_affinity[i,j]

    Cells in the same hemisphere have high spatial AND expression similarity.
    Cells across the midline have lower expression similarity (different cell
    type compositions) even when spatially adjacent.

    Parameters
    ----------
    adata      : AnnData with .obsm['spatial'] and .obs['cell_type_annot'].
    n_clusters : int -- number of clusters (2 for brain hemispheres).
    n_neighbors: int -- for kNN affinity construction.
    verbose    : bool.

    Returns
    -------
    labels : (n_cells,) int32.
    """
    import scipy.sparse as sp
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm["spatial"].astype(np.float64)
    n = len(coords)

    # ── Spatial affinity ──────────────────────────────────────────────────
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree")
    nn.fit(coords)
    dists, indices = nn.kneighbors(coords)
    dists, indices = dists[:, 1:], indices[:, 1:]
    sigma_sp = np.median(dists) + 1e-6
    W_sp = np.exp(-dists ** 2 / (2 * sigma_sp ** 2))   # (n, k)

    # ── Expression affinity (cell-type based) ─────────────────────────────
    ct = np.asarray(adata.obs["cell_type_annot"].astype(str))
    uct = np.unique(ct)
    ct_idx = np.array([np.where(uct == c)[0][0] for c in ct])

    # Expression affinity = fraction of neighbours with same cell type
    W_expr = np.zeros_like(W_sp)
    for i in range(n):
        nbrs = indices[i]
        same = (ct_idx[nbrs] == ct_idx[i]).astype(np.float32)
        W_expr[i] = same + 0.1   # +0.1 so different types still have some weight

    # ── Combined affinity ─────────────────────────────────────────────────
    W_comb = W_sp * W_expr   # element-wise product: high only when BOTH similar

    # Build sparse affinity matrix
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices.ravel()
    vals = W_comb.ravel()

    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A = (A + A.T).toarray() * 0.5   # symmetrise

    # ── Spectral clustering on combined affinity ───────────────────────────
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=42,
        n_init=10,
    )
    labels = sc.fit_predict(A).astype(np.int32)

    if verbose:
        for k in np.unique(labels):
            sz = (labels == k).sum()
            print(f"  [spectral] C_{k}: {sz} cells ({sz/n*100:.1f}%)")

    return labels


def _spectral_fallback(adata, n_components=2, verbose=True):
    from sklearn.cluster import SpectralClustering
    coords = adata.obsm["spatial"].astype(np.float64)
    if verbose:
        print(f"[BISPA] spectral clustering (K={n_components}) fallback ...")
    sc = SpectralClustering(n_clusters=n_components, affinity="nearest_neighbors",
                             n_neighbors=15, random_state=42)
    return sc.fit_predict(coords).astype(np.int32)


def _merge_small(labels, coords, min_frac):
    """Merge communities smaller than min_frac*n into nearest neighbour."""
    n = len(labels)
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        unique, counts = np.unique(labels, return_counts=True)
        for k, cnt in zip(unique, counts):
            if cnt < min_frac * n:
                mask_k = labels == k
                c_k = coords[mask_k].mean(axis=0)
                best_k2, best_d = -1, np.inf
                for k2 in unique:
                    if k2 == k:
                        continue
                    c_k2 = coords[labels == k2].mean(axis=0)
                    d = np.linalg.norm(c_k - c_k2)
                    if d < best_d:
                        best_d, best_k2 = d, k2
                if best_k2 >= 0:
                    labels[mask_k] = best_k2
                    changed = True
                    break
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[l] for l in labels], dtype=np.int32)


# =========================================================================
# Stage 2: Bipartite community matching
# =========================================================================

def _region_profile(adata, mask=None):
    """Multi-modal profile for a set of cells."""
    import scipy.sparse as sp
    if mask is not None:
        adata = adata[mask]
    ct = np.asarray(adata.obs["cell_type_annot"].astype(str))
    uct = np.unique(ct)
    cnt = np.array([(ct == c).sum() for c in uct], dtype=np.float32)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.where(np.isfinite(np.asarray(X, dtype=np.float32)), np.asarray(X, dtype=np.float32), 0.0)
    coords = adata.obsm["spatial"].astype(np.float64)
    xr = coords[:, 0].max() - coords[:, 0].min() + 1e-6
    yr = coords[:, 1].max() - coords[:, 1].min() + 1e-6
    return {
        "cell_types": uct,
        "ct_dist": cnt / cnt.sum(),
        "expr_mean": X.mean(axis=0),
        "centroid": coords.mean(axis=0),
        "aspect": xr / yr,
        "n_cells": len(adata),
    }


def _profile_dist(pA, pB, shared_ct, cross_timepoint=False):
    """
    Composite profile distance: cell-type JSD + expression cosine + aspect ratio.

    Weights: JSD=0.70, expr=0.20 (same-tp) / 0.05 (cross-tp), aspect=0.10 / 0.25.
    Cell-type JSD is the most stable signal across timepoints.
    """
    def _ct_vec(p):
        v = np.zeros(len(shared_ct), dtype=np.float64)
        for i, ct in enumerate(shared_ct):
            idx = np.where(p["cell_types"] == ct)[0]
            if len(idx):
                v[i] = p["ct_dist"][idx[0]]
        v += 1e-10
        return v / v.sum()

    va, vb = _ct_vec(pA), _ct_vec(pB)
    M = (va + vb) / 2.0
    jsd = max(float(np.sum(va * np.log(va / M) + vb * np.log(vb / M)) / 2.0), 0.0)

    ea = pA["expr_mean"].astype(np.float64)
    eb = pB["expr_mean"].astype(np.float64)
    na, nb = np.linalg.norm(ea), np.linalg.norm(eb)
    expr_d = 1.0 - float(ea @ eb) / (na * nb) if na > 1e-10 and nb > 1e-10 else 1.0

    asp_d = min(abs(np.log(max(pA["aspect"], 1e-3)) - np.log(max(pB["aspect"], 1e-3))), 1.0)

    w_jsd  = 0.70
    w_expr = 0.05 if cross_timepoint else 0.20
    w_asp  = 0.25 if cross_timepoint else 0.10
    return w_jsd * jsd + w_expr * expr_d + w_asp * asp_d


def build_community_similarity(sliceA, labels_A, sliceB, labels_B,
                               cross_timepoint=False, verbose=True):
    """
    Build the (K_A x K_B) profile-similarity matrix S[k,l].

    S[k,l] = composite distance between community k of A and community l of B.
    Lower = more similar.
    Spatial centroid proximity (10% weight) breaks bilateral symmetry ties.

    Returns S, comms_A, comms_B.
    """
    ct_A = set(sliceA.obs["cell_type_annot"].astype(str).unique())
    ct_B = set(sliceB.obs["cell_type_annot"].astype(str).unique())
    shared = np.array(sorted(ct_A & ct_B))

    comms_A = np.unique(labels_A)
    comms_B = np.unique(labels_B)
    K_A, K_B = len(comms_A), len(comms_B)

    coords_B = sliceB.obsm["spatial"].astype(np.float64)
    max_diam = float(np.linalg.norm(coords_B.max(axis=0) - coords_B.min(axis=0))) + 1e-6

    pA_list = {k: _region_profile(sliceA, labels_A == k) for k in comms_A}
    pB_list = {l: _region_profile(sliceB, labels_B == l) for l in comms_B}

    S = np.zeros((K_A, K_B), dtype=np.float32)
    for i, k in enumerate(comms_A):
        for j, l in enumerate(comms_B):
            bio_d = _profile_dist(pA_list[k], pB_list[l], shared, cross_timepoint)
            sp_d  = float(np.linalg.norm(pA_list[k]["centroid"] - pB_list[l]["centroid"])) / max_diam
            S[i, j] = bio_d + 0.10 * sp_d

    if verbose:
        print(f"[BISPA similarity] K_A={K_A}  K_B={K_B}  "
              f"S range=[{S.min():.3f}, {S.max():.3f}]")
    return S, comms_A, comms_B


def hungarian_matching(S, comms_A, comms_B, threshold=0.85, verbose=True):
    """
    Optimal 1-to-1 bipartite matching between A and B communities.

    Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm).
    Pairs with S[k,l] > threshold are declared UNMATCHED.

    If K_A != K_B, the extra communities are automatically unmatched.
    Examples:
      A has 1 community, B has 2 -> 1 matched pair, 1 unmatched B community.
      A has 2, B has 1 -> 1 matched pair, 1 unmatched A community.
      A has 2, B has 2 -> up to 2 matched pairs.

    Returns matched_pairs (list of (k_A,k_B)), unmatched_A, unmatched_B.
    """
    from scipy.optimize import linear_sum_assignment

    K_A, K_B = S.shape
    K_max = max(K_A, K_B)
    S_pad = np.full((K_max, K_max), fill_value=1e6, dtype=np.float64)
    S_pad[:K_A, :K_B] = S.astype(np.float64)

    row_ind, col_ind = linear_sum_assignment(S_pad)

    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if r >= K_A or c >= K_B:
            continue
        if S[r, c] > threshold:
            continue
        matched_pairs.append((int(comms_A[r]), int(comms_B[c])))

    matched_A = {k for k, _ in matched_pairs}
    matched_B = {l for _, l in matched_pairs}
    unmatched_A = np.array([k for k in comms_A if k not in matched_A], dtype=np.int32)
    unmatched_B = np.array([l for l in comms_B if l not in matched_B], dtype=np.int32)

    if verbose:
        print(f"[BISPA match] {len(matched_pairs)} matched pairs: {matched_pairs}")
        if len(unmatched_A):
            print(f"  Unmatched in A: {unmatched_A.tolist()} (no counterpart in B)")
        if len(unmatched_B):
            print(f"  Unmatched in B: {unmatched_B.tolist()} (receives no mass from A)")

    return matched_pairs, unmatched_A, unmatched_B


# =========================================================================
# Stage 3: Refined pose from matched community cells only
# =========================================================================

def recover_pose_matched(sliceA, labels_A, sliceB, labels_B,
                         matched_pairs, grid_size=256, verbose=True):
    """
    Estimate SE(2) pose using ONLY matched community cells.

    Running Fourier pose on full slices is confusing when one slice has
    extra unmatched communities (e.g. extra hemisphere).  By restricting
    to matched cells, both density fields represent the same anatomy.

    Algorithm:
    1. Subset both slices to matched community cells.
    2. Run estimate_pose on the subsets -> theta.
    3. Translation = size-weighted average of per-pair centroid offsets.

    Returns theta_deg, tx, ty, score.
    """
    from .pose import estimate_pose, _rotation_matrix

    if not matched_pairs:
        if verbose:
            print("[BISPA pose] No matched pairs -- theta=0, t=(0,0).")
        return 0.0, 0.0, 0.0, 0.0

    mask_A = np.isin(labels_A, [k for k, _ in matched_pairs])
    mask_B = np.isin(labels_B, [l for _, l in matched_pairs])

    if mask_A.sum() < 100 or mask_B.sum() < 100:
        if verbose:
            print("[BISPA pose] Too few matched cells -- using full slices.")
        sA_sub, sB_sub = sliceA, sliceB
    else:
        sA_sub = sliceA[mask_A]
        sB_sub = sliceB[mask_B]

    theta, _, _, score = estimate_pose(sA_sub, sB_sub, grid_size=grid_size, verbose=verbose)

    # Translation from per-pair centroid offsets
    R = _rotation_matrix(theta)
    coords_A = sliceA.obsm["spatial"].astype(np.float64)
    coords_B = sliceB.obsm["spatial"].astype(np.float64)
    cA_global = coords_A.mean(axis=0)
    coords_A_r = (R @ (coords_A - cA_global).T).T + cA_global

    tx_sum, ty_sum, w_sum = 0.0, 0.0, 0.0
    for k_A, k_B in matched_pairs:
        w = float((labels_A == k_A).sum())
        c_A_r = coords_A_r[labels_A == k_A].mean(axis=0)
        c_B   = coords_B[labels_B == k_B].mean(axis=0)
        t = c_B - c_A_r
        tx_sum += w * t[0]
        ty_sum += w * t[1]
        w_sum  += w

    tx = tx_sum / w_sum
    ty = ty_sum / w_sum

    if verbose:
        print(f"[BISPA pose] theta={theta:.1f}  tx={tx:.1f}  ty={ty:.1f}  score={score:.3f}")

    return float(theta), float(tx), float(ty), float(score)


# =========================================================================
# Stage 4a: Bidirectional anchor cost
# =========================================================================

def build_bidirectional_anchor(
    sliceA, labels_A, sliceB, labels_B,
    matched_pairs, unmatched_A, unmatched_B,
    lambda_anchor=2.0, boundary_sigma_frac=0.05,
    rho_per_cell: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    use_gpu=False, verbose=True,
):
    """
    Build the (n_A, n_B) bidirectional anchor cost matrix.

    M_anchor[i,j] encodes:
      = 0 (free)     if (community(i), community(j)) in matched_pairs
                     AND j is well inside its B community
      = lambda       if community(i) is unmatched  (those A cells go nowhere)
      = lambda       if community(j) is unmatched  (those B cells receive nothing)
      = lambda       if communities are matched but to DIFFERENT partners

        If rho_per_cell is provided, the anchor penalty is modulated by the
        per-cell relaxation weights returned by the partial-OT stage. Cells with
        lower rho are treated as more permissive, so their anchor penalty is
        reduced proportionally.

    This implements full mutual partial overlap:
    - Unmatched A cells: FUGW drops their mass (relaxed source marginal)
    - Unmatched B cells: FUGW leaves them empty (relaxed target marginal)
    - Matched cells: concentrated on their matched region with soft boundary

    Memory: (n_A, n_B) float32 = ~560 MB for 10k x 14k.
    """
    from sklearn.neighbors import NearestNeighbors

    coords_B = sliceB.obsm["spatial"].astype(np.float64)
    n_A, n_B = len(sliceA), len(sliceB)
    diam = float(np.linalg.norm(coords_B.max(axis=0) - coords_B.min(axis=0))) + 1e-6
    sigma = boundary_sigma_frac * diam + 1e-6

    rho_A_cell = rho_B_cell = None
    if rho_per_cell is not None:
        if isinstance(rho_per_cell, (tuple, list)) and len(rho_per_cell) == 2:
            rho_A_cell = np.asarray(rho_per_cell[0], dtype=np.float64)
            rho_B_cell = np.asarray(rho_per_cell[1], dtype=np.float64)
        else:
            rho_A_cell = np.asarray(rho_per_cell, dtype=np.float64)
        if rho_A_cell is not None and len(rho_A_cell) != n_A:
            raise ValueError("rho_per_cell[0] must have length n_A")
        if rho_B_cell is not None and len(rho_B_cell) != n_B:
            raise ValueError("rho_per_cell[1] must have length n_B")

    if rho_A_cell is not None:
        rho_A_scale = np.clip(rho_A_cell / (float(np.max(rho_A_cell)) + 1e-12), 0.1, 1.0).astype(np.float32)
    else:
        rho_A_scale = np.ones(n_A, dtype=np.float32)

    if rho_B_cell is not None:
        rho_B_scale = np.clip(rho_B_cell / (float(np.max(rho_B_cell)) + 1e-12), 0.1, 1.0).astype(np.float32)
    else:
        rho_B_scale = np.ones(n_B, dtype=np.float32)

    # Start fully penalised -- will clear the correct region for each matched pair
    M_anchor = np.full((n_A, n_B), lambda_anchor, dtype=np.float32)

    # Build a reverse map: for each B community, which A community is it matched to
    # match_A_to_B[k_A] = k_B
    match_A_to_B = {k: l for k, l in matched_pairs}
    set_unmatched_A = set(int(x) for x in unmatched_A)

    for k_A, k_B in matched_pairs:
        mask_A_k = labels_A == k_A
        mask_B_k = labels_B == k_B

        coords_in  = coords_B[mask_B_k]      # cells in matched B community
        coords_out = coords_B[~mask_B_k]     # cells outside (need distance to C_k_B)

        # Soft weights: cells inside C_k_B get weight 1 (free)
        # Cells near boundary get intermediate weight
        soft_B = np.zeros(n_B, dtype=np.float32)
        soft_B[mask_B_k] = 1.0   # inside: free

        if (~mask_B_k).sum() > 0 and mask_B_k.sum() > 0:
            nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
            nn.fit(coords_in)
            min_dists, _ = nn.kneighbors(coords_out)
            min_dists = min_dists.ravel()
            w_outside = np.exp(-min_dists ** 2 / (2 * sigma ** 2)).astype(np.float32)
            soft_B[~mask_B_k] = w_outside

        # For source cells in matched community k_A:
        # M_anchor[i, j] = lambda * (1 - soft_B[j])
        rows_kA = np.where(mask_A_k)[0]
        row_scale = rho_A_scale[rows_kA][:, None]
        col_scale = rho_B_scale[None, :]
        M_anchor[np.ix_(rows_kA, np.arange(n_B))] = (
            lambda_anchor * (1.0 - soft_B)[None, :] * row_scale * col_scale)

    if verbose:
        n_zero = (M_anchor == 0).sum()
        print(f"[BISPA anchor] M_anchor built: "
              f"{n_zero}/{n_A*n_B} entries free ({n_zero/(n_A*n_B)*100:.1f}%)")

    return M_anchor


# =========================================================================
# Stage 4b: Compute bidirectional marginal relaxation parameters
# =========================================================================

def compute_overlap_fractions(labels_A, labels_B, matched_pairs):
    """
    Compute per-side overlap fractions s_A and s_B.

    s_A = fraction of A cells that belong to matched communities
    s_B = fraction of B cells that belong to matched communities

    These drive the FUGW marginal relaxation:
      rho_A (source penalty) = base_reg * s_A
      rho_B (target penalty) = base_reg * s_B

    A smaller rho allows more cells to be "unmatched" on that side.
    Example:
      s_A=0.5, s_B=1.0 -> rho_A < rho_B
      -> More A cells are allowed to be unmatched than B cells.
    """
    n_A = len(labels_A)
    n_B = len(labels_B)
    if not matched_pairs:
        return 0.0, 0.0
    mask_A = np.isin(labels_A, [k for k, _ in matched_pairs])
    mask_B = np.isin(labels_B, [l for _, l in matched_pairs])
    s_A = float(mask_A.sum()) / n_A
    s_B = float(mask_B.sum()) / n_B
    return s_A, s_B


# =========================================================================
# Public main function: pairwise_align_bispa
# =========================================================================

def pairwise_align_bispa(
    sliceA,
    sliceB,
    alpha,
    beta,
    gamma,
    radius,
    filePath,
    # Decomposition
    target_min_region_frac_A=0.20,
    target_min_region_frac_B=0.20,
    leiden_resolution_A=None,
    leiden_resolution_B=None,
    min_community_size_frac=0.15,
    # Matching
    matching_threshold=0.85,
    # Anchor
    lambda_anchor=2.0,
    boundary_sigma_frac=0.05,
    # Bilateral contiguity
    lambda_spatial=0.1,
    lambda_target=0.1,
    contiguity_sigma=None,
    # FUGW
    base_reg_marginals=1.0,
    epsilon=1e-2,
    divergence="kl",
    unbalanced_solver="sinkhorn",
    max_iter_fugw=100,
    # Pose
    rough_grid_size=256,
    refined_grid_size=256,
    # cVAE (cross-timepoint)
    cvae_model=None,
    cvae_path=None,
    cvae_epochs=80,
    cvae_latent_dim=32,
    cross_timepoint=True,
    # Standard
    use_rep=None,
    numItermax=2000,
    use_gpu=False,
    gpu_verbose=True,
    verbose=False,
    sliceA_name=None,
    sliceB_name=None,
    overwrite=False,
    neighborhood_dissimilarity="jsd",
    return_diagnostics=False,
):
    """
    Bidirectional Symmetric Partial Alignment.

    This is the correct generalisation that handles ALL configurations:
    - Either slice may be the larger one with multiple regions
    - Both may be partial (mutual partial overlap)
    - Either may have symmetric sub-regions
    - Source/target assignment is arbitrary
    - Generalises to any organ without organ-specific parameters

    Parameters
    ----------
    sliceA, sliceB : AnnData -- source and target slices (roles are symmetric).
    alpha  : float -- GW spatial weight [0=biology, 1=space].
    beta   : float -- cell-type mismatch weight in M1.
    gamma  : float -- neighbourhood dissimilarity weight.
    radius : float -- neighbourhood radius (spatial coordinate units).
    filePath : str -- directory for cache files and logs.

    target_min_region_frac_A : float, default 0.20
        Adaptive Leiden target for sliceA: each community >= this fraction.
        0.20 -> K=2 for brain (two ~50% hemispheres).
        0.10 -> K<=4 for heart (four ~25% chambers).
    target_min_region_frac_B : float, default 0.20
        Same for sliceB. Can differ if slices come from different organs.
    leiden_resolution_A / leiden_resolution_B : float or None
        Override the adaptive search with a fixed Leiden resolution.
        None = adaptive (recommended).

    matching_threshold : float, default 0.85
        Maximum profile distance for a matched pair.
        Pairs above this threshold are declared unmatched.

    lambda_anchor : float, default 2.0
        Anchor cost penalty for matching outside the matched region.
    boundary_sigma_frac : float, default 0.05
        Gaussian boundary width as a fraction of sliceB's diameter.

    lambda_spatial : float, default 0.1 -- source-side contiguity weight.
    lambda_target  : float, default 0.1 -- target-side contiguity weight.

    base_reg_marginals : float, default 1.0
        Base KL marginal relaxation. Scaled per-side by overlap fractions:
          rho_A = base_reg * s_A  (where s_A = matched A cells / n_A)
          rho_B = base_reg * s_B  (where s_B = matched B cells / n_B)

    cross_timepoint : bool, default True
        If True: adjusts profile distance weights (more weight on cell-type
        JSD, less on expression); uses cVAE latent cost for M1.
        If False: standard expression cosine for M1.

    cvae_model / cvae_path : pre-trained INCENT_cVAE or path to saved model.
    cvae_epochs : int -- training epochs if training on-the-fly.

    return_diagnostics : bool, default False
        If True: returns (pi, diagnostics_dict).

    Returns
    -------
    pi : (n_A, n_B) float64 -- transport plan.
        pi.sum() < 1 indicates partial overlap was detected.

    If return_diagnostics=True:
        (pi, {
          "matched_pairs": list of (k_A,k_B),
          "unmatched_A": array,
          "unmatched_B": array,
          "s_A": float, "s_B": float,
          "labels_A": array, "labels_B": array,
          "theta_deg": float, "tx": float, "ty": float,
          "sliceA_aligned": AnnData,
          "community_similarity": S matrix,
          "pi_mass": float,
        })
    """
    import ot
    start = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_bispa_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name else f"{filePath}/log_bispa.txt")
    log = open(log_name, "w")
    log.write(f"pairwise_align_bispa -- INCENT-SE BISPA\n{datetime.datetime.now()}\n")
    log.write(f"alpha={alpha}  beta={beta}  gamma={gamma}  radius={radius}\n")
    log.write(f"lambda_anchor={lambda_anchor}  base_reg={base_reg_marginals}\n")
    log.write(f"target_frac_A={target_min_region_frac_A}  target_frac_B={target_min_region_frac_B}\n\n")

    # ------------------------------------------------------------------
    # STAGE 0: Rough rotation on full slices
    # ------------------------------------------------------------------
    print("[BISPA] Stage 0: Rough Fourier rotation ...")
    from .pose import estimate_pose, _rotation_matrix, apply_pose
    theta_rough, _, _, rough_score = estimate_pose(
        sliceA, sliceB, grid_size=rough_grid_size, verbose=gpu_verbose)
    log.write(f"Rough pose: theta={theta_rough:.1f}  score={rough_score:.3f}\n")

    # Apply rough rotation only (no translation) so spatial proximity
    # tiebreaker in Stage 2 is meaningful.
    from .rapa import apply_rotation_only_pose
    sliceA_rough = apply_rotation_only_pose(sliceA, sliceB, theta_rough, verbose=False)

    # ------------------------------------------------------------------
    # STAGE 1: Decompose BOTH slices independently
    # ------------------------------------------------------------------
    print("[BISPA] Stage 1: Decomposing sliceA ...")
    labels_A = decompose_slice(
        sliceA_rough,
        resolution=leiden_resolution_A,
        target_min_region_frac=target_min_region_frac_A,
        min_community_size_frac=min_community_size_frac,
        slice_label="A",
        verbose=gpu_verbose,
    )

    print("[BISPA] Stage 1: Decomposing sliceB ...")
    labels_B = decompose_slice(
        sliceB,
        resolution=leiden_resolution_B,
        target_min_region_frac=target_min_region_frac_B,
        min_community_size_frac=min_community_size_frac,
        slice_label="B",
        verbose=gpu_verbose,
    )

    K_A = len(np.unique(labels_A))
    K_B = len(np.unique(labels_B))
    log.write(f"Decomposition: K_A={K_A}  K_B={K_B}\n")

    # ------------------------------------------------------------------
    # STAGE 2: Bipartite community matching
    # ------------------------------------------------------------------
    print("[BISPA] Stage 2: Community similarity + Hungarian matching ...")
    S, comms_A, comms_B = build_community_similarity(
        sliceA_rough, labels_A, sliceB, labels_B,
        cross_timepoint=cross_timepoint, verbose=gpu_verbose)

    matched_pairs, unmatched_A, unmatched_B = hungarian_matching(
        S, comms_A, comms_B, threshold=matching_threshold, verbose=gpu_verbose)

    if not matched_pairs:
        print("[BISPA] WARNING: No matched pairs found. Treating as full overlap.")
        matched_pairs = [(int(comms_A[0]), int(comms_B[0]))]
        unmatched_A = comms_A[1:]
        unmatched_B = comms_B[1:]

    s_A, s_B = compute_overlap_fractions(labels_A, labels_B, matched_pairs)
    log.write(f"Matched pairs: {matched_pairs}\n")
    log.write(f"s_A={s_A:.3f}  s_B={s_B:.3f}\n")
    print(f"[BISPA] s_A={s_A:.3f}  s_B={s_B:.3f}")

    # ------------------------------------------------------------------
    # STAGE 3: Refined pose from matched community cells
    # ------------------------------------------------------------------
    print("[BISPA] Stage 3: Refined pose from matched cells ...")
    theta, tx, ty, pose_score = recover_pose_matched(
        sliceA_rough, labels_A, sliceB, labels_B,
        matched_pairs, grid_size=refined_grid_size, verbose=gpu_verbose)

    # Apply refined pose to get final aligned sliceA
    sliceA_aligned = sliceA_rough.copy()
    R = _rotation_matrix(theta)
    coords_r = sliceA_aligned.obsm["spatial"].astype(np.float64)
    cA = coords_r.mean(axis=0)
    sliceA_aligned.obsm["spatial"] = (R @ (coords_r - cA).T).T + cA + np.array([tx, ty])

    log.write(f"Refined pose: theta={theta:.1f}  tx={tx:.1f}  ty={ty:.1f}  score={pose_score:.3f}\n\n")

    # ------------------------------------------------------------------
    # STAGE 4: Cost matrices
    # ------------------------------------------------------------------
    print("[BISPA] Stage 4: Building cost matrices ...")

    # cVAE latent cost for cross-timepoint; raw cosine for same-timepoint
    model = None
    if cross_timepoint:
        from .cvae import INCENT_cVAE, train_cvae, latent_cost
        if cvae_model is not None:
            model = cvae_model
        elif cvae_path is not None and os.path.exists(cvae_path):
            model = INCENT_cVAE.load(cvae_path)
        else:
            print("[BISPA] Training cVAE ...")
            model = train_cvae([sliceA_aligned, sliceB],
                               latent_dim=cvae_latent_dim, epochs=cvae_epochs,
                               verbose=gpu_verbose)
            if cvae_path:
                model.save(cvae_path)

    # INCENT preprocessing on aligned sliceA
    from .core import _preprocess, _to_np
    log2_name = f"{filePath}/log_bispa_pre.txt"
    with open(log2_name, "w") as log2:
        p = _preprocess(
            sliceA_aligned, sliceB, alpha, beta, gamma, radius, filePath,
            use_rep, None, None, None,
            numItermax, ot.backend.NumpyBackend(), use_gpu, gpu_verbose,
            sliceA_name, sliceB_name, overwrite, neighborhood_dissimilarity,
            log2)

    M2     = p["M2"]
    D_A    = p["D_A"]
    D_B    = p["D_B"]
    a      = p["a"]
    b      = p["b"]
    sA_filt = p["sliceA"]
    sB_filt = p["sliceB"]

    D_A_np = _to_np(D_A)
    D_B_np = _to_np(D_B)
    a_np   = _to_np(a)
    b_np   = _to_np(b)
    n_A_f  = sA_filt.shape[0]
    n_B_f  = sB_filt.shape[0]

    # M1: latent cost (cross-timepoint) or cosine (same-timepoint)
    if cross_timepoint:
        from .cvae import latent_cost
        if model is None:
            raise RuntimeError("cVAE model was not initialised for cross_timepoint alignment")
        M1_np = latent_cost(sA_filt, sB_filt, model).astype(np.float64)
    else:
        M1_np = _to_np(p["cosine_dist_gene_expr"])

    # Topological fingerprint cost M_topo
    from .topology import compute_fingerprints, fingerprint_cost
    fp_A = compute_fingerprints(sA_filt, radius=radius, n_bins=16,
                                 cache_path=filePath,
                                 slice_name=f"{sliceA_name}_bispa" if sliceA_name else "A_bispa",
                                 overwrite=overwrite, verbose=gpu_verbose)
    fp_B = compute_fingerprints(sB_filt, radius=radius, n_bins=16,
                                 cache_path=filePath,
                                 slice_name=f"{sliceB_name}_bispa" if sliceB_name else "B_bispa",
                                 overwrite=overwrite, verbose=gpu_verbose)
    M_topo = fingerprint_cost(fp_A, fp_B, metric="cosine", use_gpu=use_gpu).astype(np.float64)

    # Bidirectional anchor cost
    # labels need to be remapped to the filtered slices
    def _remap_labels(labels_full, adata_full, adata_filt):
        bc_full = np.array(adata_full.obs_names)
        bc_filt = np.array(adata_filt.obs_names)
        lab_map = {bc: labels_full[i] for i, bc in enumerate(bc_full)}
        return np.array([lab_map.get(bc, -1) for bc in bc_filt], dtype=np.int32)

    labels_A_filt = _remap_labels(labels_A, sliceA_aligned, sA_filt)
    labels_B_filt = _remap_labels(labels_B, sliceB, sB_filt)

    M_anchor = build_bidirectional_anchor(
        sA_filt, labels_A_filt, sB_filt, labels_B_filt,
        matched_pairs, unmatched_A, unmatched_B,
        lambda_anchor=lambda_anchor,
        boundary_sigma_frac=boundary_sigma_frac,
        use_gpu=use_gpu, verbose=gpu_verbose)

    # Combined biological cost
    M_bio = (M1_np + 0.3 * M_topo + M_anchor).astype(np.float64)

    # FUGW alpha: (1-alpha)/alpha
    alpha_fugw = 0.0 if alpha > 1 - 1e-6 else (1e6 if alpha < 1e-6 else (1 - alpha) / alpha)

    # Per-side overlap fractions for relaxed marginals
    s_A_filt, s_B_filt = compute_overlap_fractions(labels_A_filt, labels_B_filt, matched_pairs)
    rho_A = base_reg_marginals * max(s_A_filt, 0.1)
    rho_B = base_reg_marginals * max(s_B_filt, 0.1)

    log.write(f"s_A_filt={s_A_filt:.3f}  s_B_filt={s_B_filt:.3f}\n")
    log.write(f"rho_A={rho_A:.4f}  rho_B={rho_B:.4f}  alpha_fugw={alpha_fugw:.4f}\n\n")
    print(f"[BISPA] rho_A={rho_A:.3f}  rho_B={rho_B:.3f}")

    # ------------------------------------------------------------------
    # STAGE 4: Solve partial FUGW with bidirectional relaxation
    # ------------------------------------------------------------------
    print(f"[BISPA] Solving partial FUGW (rho_A={rho_A:.3f}, rho_B={rho_B:.3f}) ...")

    pi_samp, _pif, log_dict = ot.gromov.fused_unbalanced_gromov_wasserstein(
        Cx=D_A_np,
        Cy=D_B_np,
        wx=a_np,
        wy=b_np,
        reg_marginals=(rho_A, rho_B),   # per-side relaxation
        epsilon=max(float(epsilon), 1e-2),
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha_fugw,
        M=M_bio,
        init_pi=None,
        init_duals=None,
        max_iter=max_iter_fugw,
        tol=1e-6,
        max_iter_ot=500,
        tol_ot=1e-6,
        log=True,
        verbose=verbose,
    )
    pi = np.asarray(pi_samp, dtype=np.float64)

    # Bilateral contiguity post-refinement
    if lambda_spatial > 0.0 or lambda_target > 0.0:
        print("[BISPA] Bilateral contiguity refinement ...")
        from .contiguity import contiguity_gradient, build_spatial_affinity
        from .rapa import target_contiguity_gradient, build_target_affinity
        sigma_c = contiguity_sigma if contiguity_sigma is not None else radius / 3.0
        W_A = build_spatial_affinity(sA_filt.obsm["spatial"].astype(np.float64),
                                      sigma=sigma_c, k_nn=20)
        W_B = build_target_affinity(sB_filt, sigma=sigma_c, k_nn=20)
        for _ in range(10):
            grad = np.zeros_like(pi)
            if lambda_spatial > 0.0:
                grad += lambda_spatial * contiguity_gradient(pi, W_A, D_B_np, use_gpu=use_gpu)
            if lambda_target > 0.0:
                grad += lambda_target  * target_contiguity_gradient(pi, W_B, D_A_np, use_gpu=use_gpu)
            pi = np.maximum(pi - 0.05 * grad, 0.0)
            row_sums = pi.sum(axis=1, keepdims=True)
            pi = pi / np.maximum(row_sums, 1e-12) * a_np[:, None]

    pi_mass = float(pi.sum())
    runtime = time.time() - start

    log.write(f"pi_mass={pi_mass:.4f}\nRuntime={runtime:.1f}s\n")
    log.close()

    print(f"[BISPA] Done.  pi_mass={pi_mass:.4f}  Runtime={runtime:.1f}s")

    if return_diagnostics:
        return pi, {
            "matched_pairs":        matched_pairs,
            "unmatched_A":          unmatched_A,
            "unmatched_B":          unmatched_B,
            "s_A":                  s_A_filt,
            "s_B":                  s_B_filt,
            "labels_A":             labels_A_filt,
            "labels_B":             labels_B_filt,
            "theta_deg":            theta,
            "tx":                   tx,
            "ty":                   ty,
            "sliceA_aligned":       sliceA_aligned,
            "community_similarity": S,
            "pi_mass":              pi_mass,
        }
    return pi