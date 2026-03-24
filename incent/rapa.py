"""Region-aware partial alignment helpers for INCENT-SE."""

import os
import time
import datetime
import warnings
import numpy as np
from typing import Optional, Tuple, List
from anndata import AnnData

from ._gpu import resolve_device, to_torch, to_numpy


# ═════════════════════════════════════════════════════════════════════════════
# Stage 0: Rotation-only pose for cross-timepoint data
# ═════════════════════════════════════════════════════════════════════════════

def apply_rotation_only_pose(
    sliceA: AnnData,
    sliceB: AnnData,
    theta_deg: float,
    verbose: bool = True,
) -> AnnData:
    """
    Apply ONLY the rotation from Fourier pose estimation; ignore the translation.

    Why discard the translation?
    ----------------------------
    For cross-timepoint data (different scanners/sessions), the spatial
    coordinate origins are completely unrelated.  The centroid-based translation
        t = centroid_B − R(θ) · centroid_A
    maps sliceA's centroid onto sliceB's GLOBAL centroid.  But if sliceB
    contains multiple repeated regions, its global centroid may sit between
    valid targets rather than inside one target region.  This can land sliceA
    at an ambiguous location, which disables the spatial GW term and causes
    expression-only matching.

    Instead:
      1. Rotate sliceA by θ (correct and reliable).
      2. Translate sliceA's centroid to sliceB's centroid (neutral start).
      3. Let Stage 2 region matching compute the precise translation to C*.

    Parameters
    ----------
    sliceA    : AnnData — source slice.
    sliceB    : AnnData — target slice (used only for centroid reference).
    theta_deg : float   — rotation angle from estimate_pose().
    verbose   : bool.

    Returns
    -------
    AnnData — copy of sliceA with rotated+roughly-centred coordinates.
    """
    from .pose import _rotation_matrix

    sliceA = sliceA.copy()
    R      = _rotation_matrix(theta_deg)
    coords = sliceA.obsm['spatial'].astype(np.float64)

    # Rotate around sliceA's own centroid (rotation is centroid-invariant)
    cA     = coords.mean(axis=0)
    rotated = (R @ (coords - cA).T).T + cA

    # Translate so rotated sliceA centroid sits at sliceB's centroid.
    # This is a neutral starting point; Stage 2 will correct it to C*.
    cB     = sliceB.obsm['spatial'].astype(np.float64).mean(axis=0)
    t_neutral = cB - rotated.mean(axis=0)
    sliceA.obsm['spatial'] = rotated + t_neutral

    if verbose:
        print(f"[RAPA pose] θ={theta_deg:.1f}°  neutral translation: "
              f"tx={t_neutral[0]:.1f}  ty={t_neutral[1]:.1f}")
        print(f"[RAPA pose] sliceA centred at sliceB centroid (pre region-match)")

    return sliceA


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: Unsupervised spatial decomposition of target B
# ═════════════════════════════════════════════════════════════════════════════

def decompose_target(
    sliceB: AnnData,
    n_neighbors: int = 15,
    resolution: float = None,
    min_community_size_frac: float = 0.15,
    target_min_region_frac: float = 0.20,
    use_gpu: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Unsupervised spatial decomposition of sliceB into anatomical regions.

    Algorithm
    ---------
    1. Build a spatial kNN graph on sliceB's 2D coordinates.
    2. Run Leiden community detection with **adaptive resolution**:
       - Start at a very low resolution (0.001) and increase until every
         community is smaller than target_min_region_frac of n_B.
         - This gives the coarsest valid partition — exactly the number of
         top-level anatomical regions (2 for a paired organ, 4 for a
         four-compartment organ, etc.) without over-segmenting.
    3. Merge any remaining small communities (< min_community_size_frac)
       into their spatially nearest neighbour.

    Why adaptive resolution?
    ------------------------
    A fixed resolution=0.3 produced K=13 tiny communities on a 14k-cell
    MERFISH section — each community was only ~1000 cells, far too
    small to represent a whole target region.  The problem is
    that Leiden resolution interacts with graph density in a non-linear way
    that differs across datasets and spatial scales.

    The adaptive search finds the largest resolution at which every
    community still covers at least target_min_region_frac of n_B.
    For two ~50% regions: target_min_region_frac=0.20
    gives K=2 because any finer partition creates communities < 20% of n_B.

    If resolution is passed explicitly it overrides the adaptive search.

    Parameters
    ----------
    sliceB     : AnnData — target slice.
    n_neighbors: int, default 15 — kNN graph connectivity.
    resolution : float or None
        Leiden resolution parameter.  None (default) = adaptive search.
        Pass a float to force a specific resolution (useful for ablation).
    min_community_size_frac : float, default 0.15
        After adaptive search, merge communities smaller than this fraction.
        15% prevents tiny boundary artefacts from fragmenting the partition.
    target_min_region_frac : float, default 0.20
        Adaptive search target: find the coarsest resolution where every
        community covers at least this fraction of n_B.
        0.20 → finds K=2 for two repeated regions.
        0.10 → finds K≤4 for four-way symmetry.
        Lower values allow finer partitions.
    use_gpu    : bool — kept for API consistency (community detection is CPU).
    verbose    : bool.

    Returns
    -------
    labels : (n_B,) int32 array — community index 0..K-1 for each cell.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        warnings.warn(
            "[RAPA] leidenalg / igraph not installed.  "
            "Falling back to spectral clustering.  "
            "Install with: pip install leidenalg igraph",
            stacklevel=2,
        )
        return _spectral_decompose(sliceB, verbose=verbose)

    from sklearn.neighbors import NearestNeighbors

    coords = sliceB.obsm['spatial'].astype(np.float64)
    n_B    = len(coords)

    if verbose:
        print(f"[RAPA decompose] Building spatial kNN graph (k={n_neighbors}) "
              f"on {n_B} cells …")

    # ── Build kNN graph (built once, reused for all resolution trials) ───────
    nn     = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree')
    nn.fit(coords)
    dists, indices = nn.kneighbors(coords)
    dists   = dists[:, 1:]
    indices = indices[:, 1:]

    sigma   = np.median(dists)
    weights = np.exp(-dists ** 2 / (2 * sigma ** 2)).ravel()
    rows    = np.repeat(np.arange(n_B), n_neighbors)
    cols    = indices.ravel()

    edges  = list(zip(rows.tolist(), cols.tolist()))
    G_ig   = ig.Graph(n=n_B, edges=edges, directed=False)
    G_ig.es['weight'] = weights.tolist()

    # ── Adaptive resolution search ───────────────────────────────────────────
    if resolution is not None:
        # User specified a resolution — use it directly
        res_to_use = resolution
        if verbose:
            print(f"[RAPA decompose] Using user-specified resolution={resolution}")
    else:
        # Binary search for the largest resolution where every community
        # still covers at least target_min_region_frac of n_B.
        #
        # Why this criterion?
        #   If target_min_region_frac=0.20 and we have K=2 repeated regions,
        #   each ~50% of n_B — both pass the 20% threshold.
        #   If resolution is too high and we get K=5 communities of ~20%
        #   each, SOME will be below 20% and the condition fails.
        #   We increase resolution until all communities are above the
        #   threshold, which gives us the coarsest valid partition.
        lo, hi       = 1e-4, 0.05
        best_res     = lo
        best_labels  = None

        if verbose:
            print(f"[RAPA decompose] Adaptive resolution search "
                  f"(target_min_frac={target_min_region_frac:.2f}) …")

        for _ in range(20):       # at most 20 bisection steps
            mid = (lo + hi) / 2.0
            part = leidenalg.find_partition(
                G_ig,
                leidenalg.RBConfigurationVertexPartition,
                weights='weight',
                resolution_parameter=mid,
                seed=42,
            )
            lbl   = np.array(part.membership, dtype=np.int32)
            sizes = np.array([(lbl == k).sum() for k in np.unique(lbl)])
            min_frac = sizes.min() / n_B

            if min_frac >= target_min_region_frac:
                # All communities large enough — try a higher resolution
                # (finer partition) to see if we can get more detail
                best_res    = mid
                best_labels = lbl
                lo          = mid
            else:
                # Some community too small — decrease resolution
                hi = mid

            if hi - lo < 1e-6:
                break

        if best_labels is None:
            # No resolution found all-above-threshold; use lowest tried
            if verbose:
                print(f"[RAPA decompose] Adaptive search found no valid resolution; "
                      f"falling back to spectral (K=2)")
            return _spectral_decompose(sliceB, verbose=verbose)

        res_to_use = best_res
        if verbose:
            K_found = len(np.unique(best_labels))
            print(f"[RAPA decompose] Adaptive resolution={res_to_use:.6f} "
                  f"→ K={K_found} communities")

    # ── Run final Leiden at chosen resolution ────────────────────────────────
    partition = leidenalg.find_partition(
        G_ig,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=res_to_use,
        seed=42,
    )
    labels = np.array(partition.membership, dtype=np.int32)

    # ── Merge small communities ───────────────────────────────────────────────
    labels = _merge_small_communities(labels, coords, min_community_size_frac)

    K = len(np.unique(labels))
    if verbose:
        for k in np.unique(labels):
            sz = (labels == k).sum()
            print(f"  Community {k}: {sz} cells ({sz/n_B*100:.1f}%)")
        print(f"[RAPA decompose] Found K={K} spatial communities.")

    return labels


def _spectral_decompose(sliceB: AnnData, n_components: int = 2,
                        verbose: bool = True) -> np.ndarray:
    """
    Fallback decomposition using spectral clustering on spatial coordinates.
    Used when leidenalg is not installed.
    """
    from sklearn.cluster import SpectralClustering
    coords = sliceB.obsm['spatial'].astype(np.float64)
    if verbose:
        print(f"[RAPA decompose] Spectral clustering (k={n_components}) fallback …")
    sc = SpectralClustering(n_clusters=n_components, affinity='nearest_neighbors',
                             n_neighbors=15, random_state=42)
    labels = sc.fit_predict(coords).astype(np.int32)
    return labels


def _merge_small_communities(
    labels: np.ndarray,
    coords: np.ndarray,
    min_frac: float,
) -> np.ndarray:
    """
    Merge communities smaller than min_frac * n_cells into the nearest
    community (by centroid Euclidean distance).
    """
    n      = len(labels)
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        unique, counts = np.unique(labels, return_counts=True)
        for k, cnt in zip(unique, counts):
            if cnt < min_frac * n:
                # Find centroid of this small community
                mask_k   = labels == k
                centroid_k = coords[mask_k].mean(axis=0)
                # Find centroid of every other community
                best_k, best_d = -1, np.inf
                for k2 in unique:
                    if k2 == k:
                        continue
                    centroid_k2 = coords[labels == k2].mean(axis=0)
                    d = np.linalg.norm(centroid_k - centroid_k2)
                    if d < best_d:
                        best_d, best_k = d, k2
                if best_k >= 0:
                    labels[mask_k] = best_k
                    changed = True
                    break   # restart the loop after each merge
    # Re-index labels to 0..K-1
    unique = np.unique(labels)
    remap  = {old: new for new, old in enumerate(unique)}
    return np.array([remap[l] for l in labels], dtype=np.int32)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: Source-to-region matching
# ═════════════════════════════════════════════════════════════════════════════

def _region_profile(adata: AnnData, mask: Optional[np.ndarray] = None) -> dict:
    """
    Compute a multi-modal profile for a set of cells (a region or sliceA).

    Profile components
    ------------------
    cell_type_dist : (K,) float — normalised cell-type frequency distribution.
    expression_mean: (p,) float — mean log-normalised gene expression vector.
    spatial_centroid: (2,) float — mean spatial coordinate.
    spatial_aspect : float — bounding-box aspect ratio (width/height).
        Encodes elongated shapes when repeated regions create an asymmetric
        tissue footprint.

    Parameters
    ----------
    adata : AnnData — the slice (or a view).
    mask  : optional boolean array to select a subset of cells.
    """
    import scipy.sparse as sp

    if mask is not None:
        adata = adata[mask]

    # Cell-type distribution
    ct_labels = np.asarray(adata.obs['cell_type_annot'].astype(str))
    unique_ct  = np.unique(ct_labels)
    ct_counts  = np.array([(ct_labels == ct).sum() for ct in unique_ct], dtype=np.float32)
    ct_dist    = ct_counts / ct_counts.sum()

    # Mean gene expression
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    # Handle already-normalised data: take mean directly
    expr_mean = X.mean(axis=0)

    # Spatial statistics
    coords = adata.obsm['spatial'].astype(np.float64)
    centroid = coords.mean(axis=0)
    x_range  = coords[:, 0].max() - coords[:, 0].min() + 1e-6
    y_range  = coords[:, 1].max() - coords[:, 1].min() + 1e-6
    aspect   = x_range / y_range

    return {
        'cell_types'  : unique_ct,
        'ct_dist'     : ct_dist,
        'expr_mean'   : expr_mean,
        'centroid'    : centroid,
        'aspect'      : aspect,
        'n_cells'     : len(adata),
    }


def _profile_distance(
    pA: dict,
    pB: dict,
    shared_ct: np.ndarray,
) -> float:
    """
    Composite distance between two region profiles.

    D(A, B) = w_ct   * JSD(ct_dist_A, ct_dist_B)
            + w_expr * (1 − cosine(expr_mean_A, expr_mean_B))
            + w_asp  * |log(aspect_A) − log(aspect_B)|

    Weights: cell-type JSD dominates (0.6), expression is secondary (0.3),
    shape is a weak tiebreaker (0.1).

        JSD on cell-type distributions is the strongest signal:
            - A repeated region can have roughly the same cell-type composition
                regardless of timepoint.
            - Multiple regions can look similar in expression space even when they
                occupy different compartments.
            - This means JSD alone can't distinguish repeated regions.
                We need the SPATIAL context rather than the expression-based profile.

    So the distance is used to RANK communities by similarity to A, and
    then spatial context (centroid position relative to the tissue footprint)
    breaks ties for symmetric organs.

    Parameters
    ----------
    pA, pB       : profile dicts from _region_profile().
    shared_ct    : (K,) str array — cell types present in both slices.

    Returns
    -------
    float — composite distance (lower = more similar).
    """
    # ── Cell-type JSD ──────────────────────────────────────────────────────
    def _get_ct_vec(p, shared_ct):
        """Extract cell-type distribution aligned to shared_ct."""
        v = np.zeros(len(shared_ct), dtype=np.float64)
        for i, ct in enumerate(shared_ct):
            idx = np.where(p['cell_types'] == ct)[0]
            if len(idx) > 0:
                v[i] = p['ct_dist'][idx[0]]
        v += 1e-10          # Laplace smoothing to avoid log(0)
        return v / v.sum()

    vA  = _get_ct_vec(pA, shared_ct)
    vB  = _get_ct_vec(pB, shared_ct)
    M   = (vA + vB) / 2.0
    jsd = float(np.sum(vA * np.log(vA / M) + vB * np.log(vB / M)) / 2.0)
    jsd = max(jsd, 0.0)    # numerical safety

    # ── Expression cosine distance ─────────────────────────────────────────
    ea  = pA['expr_mean'].astype(np.float64)
    eb  = pB['expr_mean'].astype(np.float64)
    # Align to same gene set (already guaranteed by preprocessing)
    na  = np.linalg.norm(ea); nb = np.linalg.norm(eb)
    if na > 1e-10 and nb > 1e-10:
        expr_cos_dist = 1.0 - float(ea @ eb) / (na * nb)
    else:
        expr_cos_dist = 1.0

    # ── Shape aspect ratio ─────────────────────────────────────────────────
    aspect_dist = abs(np.log(max(pA['aspect'], 1e-3)) -
                      np.log(max(pB['aspect'], 1e-3)))

    return 0.6 * jsd + 0.3 * expr_cos_dist + 0.1 * min(aspect_dist, 1.0)


def _spatial_side_score(
    community_centroid: np.ndarray,
    sliceA_centroid: np.ndarray,
    sliceB_coords: np.ndarray,
    community_labels: np.ndarray,
    k: int,
) -> float:
    """
    Score how well a community's SPATIAL POSITION matches sliceA's position
    relative to sliceB's overall structure.

    For symmetric organs, multiple regions can have the same expression
    profile.  The spatial tiebreaker works as follows:

    Intuition: after rotation-only pose, sliceA's centroid sits at sliceB's
    global centroid.  But local spatial context can still distinguish which
    repeated region a cell belongs to.

    This score measures: do the cells in community C_k form the half of sliceB
    that is spatially closer to sliceA (as a whole) than the other half?

    Specifically: for each cell j in C_k, compute its distance to the NEAREST
    cell in sliceA.  The community with smaller mean nearest-neighbour distance
    to sliceA is the correct one.

    This is the same principle as iterative closest point (ICP) — spatial
    proximity breaks expression symmetry.

    Parameters
    ----------
    community_centroid : (2,) float — centroid of community k.
    sliceA_centroid    : (2,) float — centroid of sliceA (post rotation-only pose).
    sliceB_coords      : (n_B, 2) float — all sliceB cell coordinates.
    community_labels   : (n_B,) int — community membership.
    k                  : int — community index.

    Returns
    -------
    float — spatial proximity score (lower = sliceA is closer to this community).
    """
    from sklearn.neighbors import NearestNeighbors

    # Distance from community centroid to sliceA centroid
    d_centroid = float(np.linalg.norm(community_centroid - sliceA_centroid))

    # Mean nearest-neighbour distance from community cells to sliceA centroid
    # (simple approximation — no sliceA cell coordinates needed here)
    # We use the centroid-to-centroid distance as the spatial score.
    # Lower is better.
    return d_centroid


def match_source_to_region(
    sliceA: AnnData,
    sliceB: AnnData,
    community_labels: np.ndarray,
    radius: float,
    verbose: bool = True,
) -> Tuple[int, np.ndarray, dict]:
    """
    Find which community in sliceB best matches sliceA.

    Uses Spatial Overlap Score: for each candidate region k, hypothetically
    place sliceA at region k's centroid and measure what fraction of sliceA
    cells find a same-cell-type neighbour within ``radius`` in region k.

    Generalises to any organ with any K regions. No new parameters.
    ``radius`` is the INCENT neighbourhood radius already required by the user.
    """
    from .region_matcher import spatial_overlap_score
    return spatial_overlap_score(
        sliceA, sliceB, community_labels, radius, verbose=verbose)


def apply_region_translation(
    sliceA: AnnData,
    region_info: dict,
    inplace: bool = False,
) -> AnnData:
    """
    Apply the fine translation that centres sliceA on the matched region C*.

    This is the second half of the SE(2) pose for cross-timepoint data:
      Phase 1 (apply_rotation_only_pose): rotation + neutral centring
      Phase 2 (this function):           fine translation to the correct region

    Parameters
    ----------
    sliceA      : AnnData — after rotation-only pose.
    region_info : dict from match_source_to_region().
    inplace     : bool.

    Returns
    -------
    AnnData with updated .obsm['spatial'].
    """
    if not inplace:
        sliceA = sliceA.copy()
    tx, ty = region_info['spatial_translation']
    sliceA.obsm['spatial'] = sliceA.obsm['spatial'].astype(np.float64) + np.array([tx, ty])
    return sliceA


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3-a: Anchor cost construction
# ═════════════════════════════════════════════════════════════════════════════

def build_anchor_cost(
    sliceB: AnnData,
    community_labels: np.ndarray,
    best_k: int,
    lambda_anchor: float = 2.0,
    boundary_sigma_frac: float = 0.05,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Build the soft anchor cost matrix M_anchor that penalises transport
    to cells outside the matched community C*.

    For cells INSIDE C*: penalty = 0 (no cost for matching here).
    For cells OUTSIDE C*: penalty = lambda_anchor (strong discouragement).
    For cells NEAR C* BOUNDARY: smooth ramp via Gaussian distance falloff.

    The boundary smoothing is critical: hard 0/1 masks create artificial
    edges at the community boundary and fragment the OT plan near the
    borders of the identified region.

    M_anchor[i, j] = lambda_anchor * (1 − w_j)
    where w_j = exp(−min_distance_j_to_C*² / (2σ²))  ∈ [0,1]
    Cells in C* have min_distance=0 → w=1 → M_anchor=0.
    Cells far from C* have large min_distance → w≈0 → M_anchor≈lambda_anchor.

    Parameters
    ----------
    sliceB           : AnnData — target slice.
    community_labels : (n_B,) int — community membership.
    best_k           : int — matched community index.
    lambda_anchor    : float, default 2.0
        Penalty for matching outside C*.  Should be comparable to the scale
        of M1+M2 (which are normalised to ~[0,1]).  2.0 = strong preference
        for C* without hard exclusion (allows a few cells to stray).
    boundary_sigma_frac : float, default 0.05
        σ = boundary_sigma_frac × diameter(sliceB).
        Controls the width of the soft boundary transition zone.
    use_gpu          : bool.

    Returns
    -------
    M_anchor : (n_A, n_B) float32 — broadcast-ready anchor cost.
        Actually returns a (1, n_B) array — identical for all cells in A.
        The calling code broadcasts it to (n_A, n_B) when adding to M1.
        This saves O(n_A × n_B) memory: the penalty only depends on j (target).
    """
    coords_B = sliceB.obsm['spatial'].astype(np.float64)
    n_B      = len(coords_B)
    in_C     = (community_labels == best_k)

    # Cells inside C* have anchor cost 0; cells outside need distance to C*
    costs = np.zeros(n_B, dtype=np.float32)

    if in_C.all() or not in_C.any():
        # Trivial cases: no anchor needed
        return costs[None, :].astype(np.float32)

    # σ = fraction of B's spatial diameter
    diam  = float(np.linalg.norm(coords_B.max(axis=0) - coords_B.min(axis=0)))
    sigma = boundary_sigma_frac * diam + 1e-6

    # For each cell OUTSIDE C*, compute its minimum distance to ANY cell in C*
    # Using batch nearest-neighbour for efficiency
    from sklearn.neighbors import NearestNeighbors

    coords_in  = coords_B[in_C]
    coords_out = coords_B[~in_C]

    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(coords_in)
    min_dists, _ = nn.kneighbors(coords_out)
    min_dists    = min_dists.ravel()

    # Smooth weight: 1 at boundary, decays to 0 away from C*
    w_boundary    = np.exp(-min_dists ** 2 / (2 * sigma ** 2)).astype(np.float32)
    # Penalty = lambda_anchor * (1 - w)
    costs[~in_C]  = lambda_anchor * (1.0 - w_boundary)
    # Cells inside C* keep cost=0

    return costs[None, :].astype(np.float32)   # (1, n_B) — broadcast to (n_A, n_B)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3-b: Target-side contiguity (bilateral regulariser)
# ═════════════════════════════════════════════════════════════════════════════

def build_target_affinity(
    sliceB: AnnData,
    sigma: float,
    k_nn: int = 20,
) -> object:
    """
    Build the sparse spatial affinity W_B for sliceB (target-side).

    W_B[j, j'] = exp(-d_B(j,j') / σ) for k-nearest neighbours.

    This is the symmetric counterpart to W_A in contiguity.py.
    Together, the bilateral regulariser:
        R_bilateral(π) = <W_A, π D_B π^T> + <W_B, π^T D_A π>
    forces BOTH the source pattern AND the target pattern to be spatially
    contiguous — i.e. nearby cells in A map to nearby cells in B AND
    nearby cells in B receive from nearby cells in A.

    Parameters
    ----------
    sliceB : AnnData — target slice.
    sigma  : float   — affinity decay length (same units as spatial coords).
    k_nn   : int     — number of nearest neighbours.

    Returns
    -------
    scipy.sparse.csr_matrix — (n_B, n_B) symmetric float32.
    """
    from .contiguity import build_spatial_affinity
    return build_spatial_affinity(
        sliceB.obsm['spatial'].astype(np.float64), sigma=sigma, k_nn=k_nn)


def target_contiguity_gradient(
    pi: np.ndarray,
    W_B: object,
    D_A: np.ndarray,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Gradient of R_B(π) = <W_B, π^T D_A π> w.r.t. π.

    Derivation (symmetric to the source-side contiguity):
        ∂R_B/∂π = 2 · D_A · π · W_B

    Parameters
    ----------
    pi      : (n_A, n_B) float — transport plan.
    W_B     : (n_B, n_B) scipy CSR — target spatial affinity.
    D_A     : (n_A, n_A) float — pairwise distances in A.
    use_gpu : bool.

    Returns
    -------
    grad : (n_A, n_B) float64.
    """
    device = resolve_device(use_gpu)

    if device == 'cuda':
        import torch
        pi_t   = to_torch(pi,  device, dtype=torch.float32)
        D_A_t  = to_torch(D_A, device, dtype=torch.float32)
        from ._gpu import sparse_to_torch
        W_B_t  = sparse_to_torch(W_B, device, dtype=torch.float32)

        # D_A · π : (n_A, n_A) @ (n_A, n_B) = (n_A, n_B)
        DA_pi  = D_A_t @ pi_t
        # (D_A · π) · W_B : (n_A, n_B) @ sparse(n_B, n_B) = (n_A, n_B)
        grad   = torch.mm(DA_pi, W_B_t.to_dense())   # sparse matmul output
        return (2.0 * grad).cpu().numpy().astype(np.float64)

    # ── CPU ────────────────────────────────────────────────────────────────
    DA_pi  = D_A @ pi
    grad   = DA_pi @ W_B
    return 2.0 * np.asarray(grad, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: pairwise_align_rapa — the full pipeline
# ═════════════════════════════════════════════════════════════════════════════

def pairwise_align_rapa(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    filePath: str,
    # ── Pose ───────────────────────────────────────────────────────────────
    theta_deg: Optional[float] = None,
    estimate_rotation: bool = True,
    rotation_only_pose: bool = True,
    # ── Decomposition ──────────────────────────────────────────────────────
    leiden_resolution: float = None,
    target_min_region_frac: float = 0.20,
    min_community_size_frac: float = 0.15,
    # ── Anchor ─────────────────────────────────────────────────────────────
    lambda_anchor: float = 2.0,
    boundary_sigma_frac: float = 0.05,
    # ── Bilateral contiguity ───────────────────────────────────────────────
    lambda_spatial: float = 0.1,
    lambda_target: float = 0.1,
    contiguity_sigma: Optional[float] = None,
    # ── FUGW unbalanced solver ─────────────────────────────────────────────
    reg_marginals: float = 1.0,
    epsilon: float = 0.0,
    divergence: str = 'kl',
    unbalanced_solver: str = 'mm',
    max_iter_fugw: int = 100,
    # ── cVAE for cross-timepoint ────────────────────────────────────────────
    cvae_model=None,
    cvae_path: Optional[str] = None,
    cvae_epochs: int = 80,
    cvae_latent_dim: int = 32,
    # ── Standard INCENT params ─────────────────────────────────────────────
    use_rep: Optional[str] = None,
    numItermax: int = 2000,
    use_gpu: bool = False,
    gpu_verbose: bool = True,
    verbose: bool = False,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool = False,
    neighborhood_dissimilarity: str = 'jsd',
    return_diagnostics: bool = False,
    **kwargs,
):
    """
    Region-Aware Partial Alignment (RAPA).

    Solves the full problem: sliceA is an unknown sub-region of sliceB,
    which may contain multiple anatomically symmetric regions.  Both
    cross-timepoint (different coordinate frames) and same-timepoint
    cases are handled.

    Pipeline
    --------
    Stage 0: Rotation-only pose (or full SE(2) if same-timepoint)
    Stage 1: Spatial decomposition of sliceB into K communities
    Stage 2: Match sliceA to best community C*; recover fine translation
    Stage 3: Soft-anchored unbalanced FUGW with bilateral contiguity

    Parameters
    ----------
    sliceA, sliceB : AnnData — source and target slices.
    alpha  : float — GW spatial weight (0=biology, 1=space).
    beta   : float — cell-type mismatch weight in M1.
    gamma  : float — neighbourhood dissimilarity weight.
    radius : float — neighbourhood radius (spatial units).
    filePath: str  — directory for cache files and logs.

    theta_deg : float or None
        Pre-computed rotation angle from estimate_pose().
        If None and estimate_rotation=True, it is computed here.
    estimate_rotation : bool, default True
        Run Fourier rotation estimation if theta_deg is not provided.
    rotation_only_pose : bool, default True
        If True, use rotation-only pose (discard translation from Fourier).
        Set False for same-timepoint data where the translation is valid.

    leiden_resolution : float, default 0.3
        Controls the granularity of community detection.
        0.1–0.3: coarse (paired regions, major lobes).
        0.5–1.0: fine (cortical layers, lobules).
    min_community_size_frac : float, default 0.05
        Merge communities smaller than this fraction of n_B.

    lambda_anchor : float, default 2.0
        Penalty weight for transport outside the matched community C*.
        ~2x the scale of the biological cost (which is in [0,1]).
    boundary_sigma_frac : float, default 0.05
        Width of the soft boundary transition zone (fraction of B's diameter).

    lambda_spatial : float, default 0.1 — source-side contiguity weight.
    lambda_target  : float, default 0.1 — target-side contiguity weight.

    reg_marginals : float, default 1.0
        KL penalty on marginal violations in FUGW.
        Lower → more partial matching allowed.
        s_prior is used to scale this automatically.
    epsilon : float, default 0.0 — Sinkhorn regularisation (0 = exact MM solver).

    cvae_model / cvae_path : pre-trained cVAE for cross-timepoint expression cost.
    cvae_epochs : int — epochs for on-the-fly cVAE training.

    return_diagnostics : bool, default False
        If True, return additional diagnostic information.

    Returns
    -------
    pi : (n_A, n_B) float64 — FUGW transport plan.
        pi.sum() < 1 indicates partial overlap was detected.

    If return_diagnostics=True:
        (pi, diagnostics_dict) where diagnostics contains:
        - 'matched_region': best_k community index
        - 'overlap_fraction': s = pi.sum()
        - 'region_scores': composite distances to all communities
        - 'theta_deg': rotation angle used
        - 'sliceA_aligned': AnnData with final aligned coordinates
        - 'community_labels': (n_B,) community assignment array
    """
    import ot
    start = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_rapa_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name else f"{filePath}/log_rapa.txt")
    logFile  = open(log_name, "w")
    logFile.write("pairwise_align_rapa — INCENT-SE RAPA\n")
    logFile.write(f"{datetime.datetime.now()}\n")
    logFile.write(f"alpha={alpha}  beta={beta}  gamma={gamma}  radius={radius}\n")
    logFile.write(f"lambda_anchor={lambda_anchor}  leiden_resolution={leiden_resolution}\n\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 0: Pose
    # ═══════════════════════════════════════════════════════════════════════
    if theta_deg is None and estimate_rotation:
        from .pose import estimate_pose
        print("[RAPA] Stage 0: Fourier rotation estimation …")
        theta_deg, _, _, pscore = estimate_pose(
            sliceA, sliceB, grid_size=256, verbose=gpu_verbose)
        logFile.write(f"Fourier pose: θ={theta_deg:.2f}°  score={pscore:.3f}\n")
    elif theta_deg is None:
        theta_deg = 0.0
        logFile.write("No rotation applied (theta_deg=0)\n")

    if rotation_only_pose:
        print(f"[RAPA] Stage 0: Rotation-only pose θ={theta_deg:.1f}° …")
        sliceA_aligned = apply_rotation_only_pose(
            sliceA, sliceB, theta_deg, verbose=gpu_verbose)
    else:
        from .pose import apply_pose, estimate_pose
        _, tx, ty, _ = estimate_pose(sliceA, sliceB, grid_size=256, verbose=False)
        sliceA_aligned = apply_pose(sliceA, theta_deg, tx, ty, inplace=False)
        print(f"[RAPA] Stage 0: Full SE(2) pose θ={theta_deg:.1f}° tx={tx:.1f} ty={ty:.1f}")

    logFile.write(f"Rotation-only pose: θ={theta_deg:.2f}°  mode={'rotation_only' if rotation_only_pose else 'full_SE2'}\n\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1: Target decomposition
    # ═══════════════════════════════════════════════════════════════════════
    print("[RAPA] Stage 1: Spatial decomposition of target …")
    community_labels = decompose_target(
        sliceB,
        resolution=leiden_resolution,
        target_min_region_frac=target_min_region_frac,
        min_community_size_frac=min_community_size_frac,
        verbose=gpu_verbose,
    )
    K = len(np.unique(community_labels))
    logFile.write(f"Decomposition: K={K} communities\n")
    for k in np.unique(community_labels):
        n_k = (community_labels == k).sum()
        logFile.write(f"  C_{k}: {n_k} cells ({n_k/len(sliceB)*100:.1f}%)\n")
    logFile.write("\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2: Source-to-region matching + fine translation
    # ═══════════════════════════════════════════════════════════════════════
    print("[RAPA] Stage 2: Matching source to best target region …")
    best_k, region_scores, region_info = match_source_to_region(
        sliceA_aligned, sliceB, community_labels,
        radius=radius,
        verbose=gpu_verbose)
    # Apply the fine translation to centre sliceA on C*
    sliceA_aligned = apply_region_translation(sliceA_aligned, region_info)
    overlap_prior  = region_info['overlap_fraction']

    logFile.write(f"Matched region: C_{best_k}  n={region_info['n_cells']}  "
                  f"s_prior={overlap_prior:.3f}\n")
    logFile.write(f"Fine translation: {region_info['spatial_translation']}\n\n")

    print(f"[RAPA] Matched C_{best_k} (n={region_info['n_cells']}, "
          f"s≈{overlap_prior:.3f})")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 3: Soft-anchored unbalanced FUGW
    # ═══════════════════════════════════════════════════════════════════════
    print("[RAPA] Stage 3: Building cost matrices …")

    # ── cVAE latent cost for cross-timepoint expression ────────────────────
    from .core import _preprocess, _to_np
    from .cvae import INCENT_cVAE, train_cvae, latent_cost

    if cvae_model is not None:
        model = cvae_model
    elif cvae_path is not None and os.path.exists(cvae_path):
        model = INCENT_cVAE.load(cvae_path)
    else:
        print("[RAPA] Training cVAE …")
        model = train_cvae([sliceA_aligned, sliceB],
                           latent_dim=cvae_latent_dim, epochs=cvae_epochs,
                           verbose=gpu_verbose)
        if cvae_path:
            model.save(cvae_path)

    # ── INCENT preprocessing (shared genes, distances, JSD, marginals) ────
    print("[RAPA] Stage 3: INCENT preprocessing …")
    import io
    with open(os.devnull, 'w') as dummy_log:
        # We need a file-like object — use a real file
        log_pre_name = f"{filePath}/log_rapa_pre_{sliceA_name}_{sliceB_name}.txt"
        with open(log_pre_name, 'w') as logPreFile:
            p = _preprocess(
                sliceA_aligned, sliceB, alpha, beta, gamma, radius, filePath,
                use_rep, None, None, None,
                numItermax, ot.backend.NumpyBackend(), use_gpu, gpu_verbose,
                sliceA_name, sliceB_name, overwrite, neighborhood_dissimilarity,
                logPreFile,
            )

    nx     = p['nx']
    M2     = p['M2']
    D_A    = p['D_A']
    D_B    = p['D_B']
    a      = p['a']
    b      = p['b']
    sA     = p['sliceA']
    sB     = p['sliceB']

    # Latent cost replaces M1 for cross-timepoint
    M_latent_np  = latent_cost(sA, sB, model)   # float32
    # Topology fingerprint cost
    from .topology import compute_fingerprints, fingerprint_cost
    from .core_se import _to_np as cse_to_np
    fp_A = compute_fingerprints(sA, radius=radius, n_bins=16,
                                 cache_path=filePath,
                                 slice_name=f"{sliceA_name}_rapa" if sliceA_name else "A_rapa",
                                 overwrite=overwrite, verbose=gpu_verbose)
    fp_B = compute_fingerprints(sB, radius=radius, n_bins=16,
                                 cache_path=filePath,
                                 slice_name=f"{sliceB_name}_rapa" if sliceB_name else "B_rapa",
                                 overwrite=overwrite, verbose=gpu_verbose)
    from .topology import fingerprint_cost as fp_cost
    M_topo_np = fp_cost(fp_A, fp_B, metric='cosine', use_gpu=use_gpu)

    # Anchor cost — the key new component
    # community_labels may refer to sliceB before filtering by shared cell types.
    # Remap labels to filtered sliceB (sB).
    sB_barcodes   = sB.obs_names
    full_barcodes = sliceB.obs_names
    label_map     = {bc: community_labels[i] for i, bc in enumerate(full_barcodes)}
    comm_labels_filt = np.array([label_map.get(bc, -1) for bc in sB_barcodes], dtype=np.int32)
    M_anchor_row = build_anchor_cost(sB, comm_labels_filt, best_k,
                                      lambda_anchor=lambda_anchor,
                                      boundary_sigma_frac=boundary_sigma_frac,
                                      use_gpu=use_gpu)   # (1, n_B)

    # Combine into total biological cost
    # M_bio = M_latent + gamma*M_topo + M_anchor (broadcast)
    M_bio_np = (M_latent_np.astype(np.float32)
                + 0.3 * M_topo_np.astype(np.float32)
                + M_anchor_row)                           # (n_A, n_B) after broadcast
    M_bio_np = M_bio_np.astype(np.float64)

    # Spatial distance matrices
    D_A_np = _to_np(D_A)
    D_B_np = _to_np(D_B)
    a_np   = _to_np(a)
    b_np   = _to_np(b)
    n_A, n_B = sA.shape[0], sB.shape[0]

    # FUGW alpha parameter: (1-alpha)/alpha converts INCENT convention to POT
    if alpha < 1e-6:
        alpha_fugw = 1e6
    elif alpha > 1.0 - 1e-6:
        alpha_fugw = 0.0
    else:
        alpha_fugw = (1.0 - alpha) / alpha

    logFile.write(f"alpha_fugw={alpha_fugw:.4f}  s_prior={overlap_prior:.3f}\n")
    logFile.write(f"M_anchor: lambda={lambda_anchor}  "
                  f"cells_in_C*={comm_labels_filt.sum() if hasattr(comm_labels_filt,'sum') else 'N/A'}\n\n")

    # ── Contiguity affinities (source + target bilateral) ──────────────────
    sigma_c = contiguity_sigma if contiguity_sigma is not None else radius / 3.0
    from .contiguity import (build_spatial_affinity, contiguity_gradient,
                              augment_fgw_gradient)
    W_A = build_spatial_affinity(sA.obsm['spatial'].astype(np.float64),
                                  sigma=sigma_c, k_nn=20)
    W_B = build_target_affinity(sB, sigma=sigma_c, k_nn=20)

    # ── Run FUGW ──────────────────────────────────────────────────────────
    print(f"[RAPA] Stage 3: Solving partial FUGW "
          f"(s_prior={overlap_prior:.3f}, reg_marginals={reg_marginals}) …")

    pi_samp, _pi_feat, log_dict = ot.gromov.fused_unbalanced_gromov_wasserstein(
        Cx=D_A_np,
        Cy=D_B_np,
        wx=a_np,
        wy=b_np,
        reg_marginals=reg_marginals * overlap_prior,  # scale penalty by expected overlap
        epsilon=epsilon,
        divergence=divergence,
        unbalanced_solver=unbalanced_solver,
        alpha=alpha_fugw,
        M=M_bio_np,
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

    # ── Bilateral contiguity post-refinement ──────────────────────────────
    if lambda_spatial > 0.0 or lambda_target > 0.0:
        print("[RAPA] Bilateral contiguity refinement …")
        for _ in range(10):
            grad_src  = (lambda_spatial *
                         contiguity_gradient(pi, W_A, D_B_np, use_gpu=use_gpu)
                         if lambda_spatial > 0.0
                         else 0.0)
            grad_tgt  = (lambda_target *
                         target_contiguity_gradient(pi, W_B, D_A_np, use_gpu=use_gpu)
                         if lambda_target > 0.0
                         else 0.0)
            grad = (grad_src if not isinstance(grad_src, float) else 0.0) + \
                   (grad_tgt if not isinstance(grad_tgt, float) else 0.0)
            if not isinstance(grad, np.ndarray):
                break
            pi = np.maximum(pi - 0.05 * grad, 0.0)
            row_sums = pi.sum(axis=1, keepdims=True)
            pi = pi / np.maximum(row_sums, 1e-12) * a_np[:, None]

    pi_mass = float(pi.sum())
    logFile.write(f"FUGW pi_mass={pi_mass:.4f}\n")
    logFile.write(f"Runtime: {time.time()-start:.1f}s\n")
    logFile.close()

    print(f"[RAPA] Done.  pi_mass={pi_mass:.4f}  "
          f"Runtime={time.time()-start:.1f}s")

    if return_diagnostics:
        diag = {
            'matched_region'   : best_k,
            'overlap_fraction' : pi_mass,
            'region_scores'    : region_scores,
            'community_labels' : community_labels,
            'theta_deg'        : theta_deg,
            'sliceA_aligned'   : sliceA_aligned,
            'region_info'      : region_info,
        }
        return pi, diag

    return pi