"""Region matching helpers for repeated anatomical regions."""

import numpy as np
from typing import List, Tuple, Optional
from anndata import AnnData
from sklearn.neighbors import BallTree


def spatial_overlap_score(
    sliceA: AnnData,
    sliceB: AnnData,
    community_labels: np.ndarray,
    radius: float,
    verbose: bool = True,
) -> Tuple[int, np.ndarray, dict]:
    """
    Find the region of sliceB that best matches sliceA using Spatial Overlap Score.

    This function replaces ``match_source_to_region`` in rapa.py.
    Organ-agnostic: works for any organ with repeated or symmetric regions.

    For each candidate region k:
      1. Tentatively place sliceA centred on region k (hypothetical translation).
      2. For each sliceA cell i: count same-cell-type neighbours in region k
         within neighbourhood radius r.
      3. overlap_score(k) = fraction of sliceA cells that found at least one
         same-type neighbour (geometric AND biological compatibility).

    Score is high only when:
      (a) sliceA geometrically overlaps region k (spatial test)
      (b) cell types are compatible (biological test)
    Both must hold simultaneously → only the CORRECT region wins.

    Parameters
    ----------
    sliceA           : AnnData with .obsm['spatial'] and .obs['cell_type_annot']
    sliceB           : AnnData with .obsm['spatial'] and .obs['cell_type_annot']
    community_labels : (n_B,) int32 from decompose_target() or decompose_slice()
    radius           : float — INCENT neighbourhood radius (same units as coords)
    verbose          : bool

    Returns
    -------
    best_k      : int
    scores      : (K,) float — overlap score per region (higher = better)
    region_info : dict
    """
    coords_A = sliceA.obsm['spatial'].astype(np.float64)
    coords_B = sliceB.obsm['spatial'].astype(np.float64)
    types_A  = np.asarray(sliceA.obs['cell_type_annot'].astype(str))
    types_B  = np.asarray(sliceB.obs['cell_type_annot'].astype(str))

    centroid_A  = coords_A.mean(axis=0)
    communities = np.unique(community_labels)
    K           = len(communities)
    n_A         = len(coords_A)

    scores    = np.zeros(K, dtype=np.float64)
    centroids = np.zeros((K, 2), dtype=np.float64)

    if verbose:
        print(f"[RegionMatch] Spatial overlap scoring: "
              f"n_A={n_A}  K={K}  radius={radius:.1f}")

    for ki, k in enumerate(communities):
        mask_k     = community_labels == k
        coords_Bk  = coords_B[mask_k]
        types_Bk   = types_B[mask_k]
        n_k        = int(mask_k.sum())
        centroid_k = coords_Bk.mean(axis=0)
        centroids[ki] = centroid_k

        if n_k == 0:
            continue

        # Tentative translation: sliceA centroid → region k centroid
        t_k          = centroid_k - centroid_A
        coords_A_try = coords_A + t_k

        # BallTree on region k cells for radius queries
        tree_k    = BallTree(coords_Bk)
        nbr_lists = tree_k.query_radius(coords_A_try, r=radius)

        cell_scores = np.zeros(n_A, dtype=np.float64)
        for i in range(n_A):
            nbrs = nbr_lists[i]
            if len(nbrs) == 0:
                cell_scores[i] = 0.0
                continue
            same = (types_Bk[nbrs] == types_A[i]).sum()
            cell_scores[i] = float(same) / float(len(nbrs))

        scores[ki] = float(cell_scores.mean())

        if verbose:
            frac_covered = float((cell_scores > 0).mean())
            print(f"  Region {k}: n={n_k:5d}  "
                  f"overlap_score={scores[ki]:.4f}  "
                  f"covered={frac_covered*100:.0f}%")

    best_ki       = int(np.argmax(scores))
    best_k        = int(communities[best_ki])
    mask_best     = community_labels == best_k
    n_best        = int(mask_best.sum())
    centroid_best = centroids[best_ki]
    ranked_ks, ranked_scores, ranked_weights = rank_region_candidates(
        scores, communities, top_k=min(3, len(communities)))

    spatial_translation = centroid_best - centroid_A
    overlap_fraction    = min(1.0, float(n_A) / max(n_best, 1))
    score_gap = float(scores[best_ki] - np.partition(scores, -2)[-2]) if len(scores) >= 2 else float(scores[best_ki])

    region_info = {
        'best_k'              : best_k,
        'centroid'            : centroid_best,
        'n_cells'             : n_best,
        'spatial_translation' : spatial_translation,
        'overlap_fraction'    : overlap_fraction,
        'all_scores'          : scores,
        'communities'         : communities,
        'ranked_communities'   : ranked_ks,
        'ranked_scores'        : ranked_scores,
        'ranked_weights'       : ranked_weights,
        'score_gap'            : score_gap,
    }

    if verbose:
        print(f"[RegionMatch] Winner: region {best_k}  "
              f"(score={scores[best_ki]:.4f}  n={n_best}  "
              f"s≈{overlap_fraction:.3f})")
        print(f"[RegionMatch] Translation: "
              f"Δx={spatial_translation[0]:.1f}  "
              f"Δy={spatial_translation[1]:.1f}")

    return best_k, scores, region_info


def rank_region_candidates(
    scores: np.ndarray,
    communities: np.ndarray,
    top_k: int = 3,
    temperature: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank candidate regions and return a softmax over the top candidates."""
    scores = np.asarray(scores, dtype=np.float64)
    communities = np.asarray(communities, dtype=np.int32)
    if scores.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    order = np.argsort(scores)[::-1]
    order = order[:max(1, min(top_k, len(order)))]
    ranked_ks = communities[order]
    ranked_scores = scores[order]

    temp = max(float(temperature), 1e-6)
    logits = (ranked_scores - ranked_scores.max()) / temp
    weights = np.exp(logits)
    weights /= max(weights.sum(), 1e-12)
    return ranked_ks, ranked_scores, weights


def _build_community_overlap_matrix(
    sliceA: AnnData,
    labels_A: np.ndarray,
    sliceB: AnnData,
    labels_B: np.ndarray,
    radius: float,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a (K_A × K_B) spatial overlap similarity matrix for BISPA matching.

    For each pair of communities (k_A from sliceA, k_B from sliceB):
      S[k_A, k_B] = spatial_overlap_score between community k_A of sliceA
                    and community k_B of sliceB.

    This replaces build_community_similarity in bispa.py for the matching step.
    Hungarian matching on (1 - S) gives the optimal community correspondence.

    ORGAN-GENERALITY:
    K_A = K_B = 2 → symmetric paired-region matching
      K_A = K_B = 4 → heart chamber matching
      K_A != K_B    → mutual partial overlap (one slice has extra regions)

    Parameters
    ----------
    sliceA, labels_A : source slice and its community labels
    sliceB, labels_B : target slice and its community labels
    radius           : float — INCENT neighbourhood radius
    verbose          : bool

    Returns
    -------
    S        : (K_A, K_B) float — overlap similarity matrix (higher = better)
    comms_A  : (K_A,) int — community indices of sliceA
    comms_B  : (K_B,) int — community indices of sliceB
    """
    coords_A = sliceA.obsm['spatial'].astype(np.float64)
    coords_B = sliceB.obsm['spatial'].astype(np.float64)
    types_A  = np.asarray(sliceA.obs['cell_type_annot'].astype(str))
    types_B  = np.asarray(sliceB.obs['cell_type_annot'].astype(str))

    comms_A = np.unique(labels_A)
    comms_B = np.unique(labels_B)
    K_A, K_B = len(comms_A), len(comms_B)

    S = np.zeros((K_A, K_B), dtype=np.float64)

    if verbose:
        print(f"[RegionMatch] Community overlap matrix: "
              f"K_A={K_A}  K_B={K_B}  radius={radius:.1f}")

    for ia, ka in enumerate(comms_A):
        mask_A = labels_A == ka
        cA_sub = coords_A[mask_A]
        tA_sub = types_A[mask_A]
        ctr_A  = cA_sub.mean(axis=0)
        n_A_k  = len(cA_sub)

        for ib, kb in enumerate(comms_B):
            mask_B = labels_B == kb
            cB_sub = coords_B[mask_B]
            tB_sub = types_B[mask_B]
            n_B_k  = len(cB_sub)

            if n_A_k == 0 or n_B_k == 0:
                continue

            # Tentative translation: community k_A centroid → community k_B centroid
            ctr_B    = cB_sub.mean(axis=0)
            t_kk     = ctr_B - ctr_A
            cA_try   = cA_sub + t_kk

            tree_B = BallTree(cB_sub)
            nbr_lists = tree_B.query_radius(cA_try, r=radius)

            cell_scores = np.zeros(n_A_k, dtype=np.float64)
            for i in range(n_A_k):
                nbrs = nbr_lists[i]
                if len(nbrs) == 0:
                    continue
                same = (tB_sub[nbrs] == tA_sub[i]).sum()
                cell_scores[i] = float(same) / float(len(nbrs))

            S[ia, ib] = float(cell_scores.mean())

    if verbose:
        print(f"[RegionMatch] S range: [{S.min():.3f}, {S.max():.3f}]")

    return S, comms_A, comms_B


def compute_region_spatial_prior(
    sliceA_placed: AnnData,
    sliceB: AnnData,
    community_labels: np.ndarray,
    radius: float,
    best_k: Optional[int] = None,
    best_ks: Optional[np.ndarray] = None,
    region_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-cell spatial prior weight vector for sliceB.

    Cells inside the winning region: weight = 1.0
    Cells at the region boundary: smooth Gaussian decay over one radius
    Cells far outside: floor weight = 0.01

    The transition width is set to ``radius`` — the same neighbourhood
    scale already used throughout INCENT. No new parameters.

    Parameters
    ----------
    sliceA_placed    : AnnData — sliceA at the correct region position
    sliceB           : AnnData
    community_labels : (n_B,) int
    best_k           : int — winning region index
    radius           : float — INCENT neighbourhood radius

    Returns
    -------
    w_prior : (n_B,) float64 in [0.01, 1.0]
    """
    coords_B = sliceB.obsm['spatial'].astype(np.float64)
    n_B      = len(coords_B)
    def _single_region_prior(region_k: int) -> np.ndarray:
        mask_in = (community_labels == region_k)
        w_prior = np.where(mask_in, 1.0, 0.01).astype(np.float64)

        n_in = int(mask_in.sum())
        n_out = int((~mask_in).sum())
        if n_in > 0 and n_out > 0:
            coords_in = coords_B[mask_in]
            coords_out = coords_B[~mask_in]
            tree_in = BallTree(coords_in)
            dist_bd, _ = tree_in.query(coords_out, k=1)
            dist_bd = dist_bd.ravel()
            sigma = radius
            w_out = 0.01 + 0.49 * np.exp(-0.5 * (dist_bd / sigma) ** 2)
            w_prior[~mask_in] = w_out
        return w_prior

    if best_ks is None:
        if best_k is None:
            raise ValueError("Either best_k or best_ks must be provided")
        return _single_region_prior(int(best_k))

    best_ks_arr = np.asarray(best_ks, dtype=np.int32)
    if region_weights is None:
        region_weights_arr = np.ones(len(best_ks_arr), dtype=np.float64)
    else:
        region_weights_arr = np.asarray(region_weights, dtype=np.float64)
    if region_weights_arr.size != len(best_ks_arr):
        raise ValueError("region_weights must match best_ks length")
    region_weights_arr = np.maximum(region_weights_arr, 0.0)
    region_weights_arr = region_weights_arr / max(region_weights_arr.sum(), 1e-12)

    blended = np.full(len(coords_B), 0.01, dtype=np.float64)
    for region_k, weight in zip(best_ks_arr, region_weights_arr):
        blended = np.maximum(blended, weight * _single_region_prior(int(region_k)))
    blended = np.clip(blended, 0.01, 1.0)
    return blended
