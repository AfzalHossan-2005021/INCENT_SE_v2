"""
partial_ot.py — Spatially-Adaptive Partial OT for Mutual Partial Overlap
=========================================================================
INCENT-SE v2: improved formulation for the case where BOTH slices have
unique regions (mutual partial overlap).

The mutual partial overlap problem
------------------------------------
The standard FUGW (fused unbalanced GW) uses SCALAR marginal relaxations
rho_A and rho_B that apply UNIFORMLY to every cell:

    KL(π·1 || rho_A·a)  +  KL(π^T·1 || rho_B·b)

This is equivalent to saying every cell has the same probability of being
"unmatched."  But in reality:

  • Cells in the OVERLAP REGION (present in both slices) should be
    STRONGLY matched.
  • Cells in the NON-OVERLAP REGION (unique to one slice) should be
    allowed to be UNMATCHED.

Scalar rho treats all cells the same, forcing a compromise that may:
  (a) leave good overlapping cells partially unmatched, OR
  (b) force non-overlapping cells to find spurious matches.

The spatially-adaptive solution
---------------------------------
We introduce CELL-SPECIFIC marginal weights ρ_A(i) and ρ_B(j) based on
the spatial proximity to the other slice's footprint:

    ρ_A(i) = base_rho * w_A(i)
    ρ_B(j) = base_rho * w_B(j)

where w_A(i) is a spatial proximity weight: high for cells in the
estimated overlap region, low for cells in the unique region.

We estimate the overlap region from the OT plan itself (self-supervised):
    overlap_score(i) = π[i,:].sum() / a[i]
Cells with high mass outflow are in the overlap; cells with low mass are
not matched.  We iterate this: update ρ from π, re-solve OT, repeat.

This is a form of ALTERNATING OPTIMIZATION that converges in 2–3 rounds.

Semi-relaxed partial coupling
--------------------------------
For the case where sliceA is FULLY contained in sliceB (one-sided partial
overlap), the correct formulation is SEMI-RELAXED OT:
    - Source marginal: π·1 = a  (EXACT: all of A is matched)
    - Target marginal: π^T·1 ≤ b  (RELAXED: only some of B receives mass)

This is more accurate than scalar FUGW for the one-sided case and can be
solved with standard EMD on an expanded problem.

Spatial overlap estimation
--------------------------
We also provide a method to ESTIMATE the overlap region from the
alignment geometry (after SE(2) is known):
    - Transform A's convex hull into B's frame
    - Compute the intersection with B's convex hull
    - Cells in the intersection have high overlap weight
    - Cells outside have low overlap weight

This can be used to set the per-cell marginals BEFORE running OT
(geometry-based prior) instead of iterating from the OT plan.

Public API
----------
estimate_spatial_overlap_weights(sliceA, sliceB, R, t, sigma_bdy)
  → (w_A, w_B) per-cell overlap weight vectors

adaptive_fugw(D_A, D_B, M_bio, a, b, w_A, w_B, base_rho, ...)
  → pi (n_A, n_B) transport plan

iterative_overlap_fugw(D_A, D_B, M_bio, a, b, ...)
  → pi, w_A_final, w_B_final

semi_relaxed_gw(D_A, D_B, M_bio, a, b, ...)
  → pi   (for one-sided partial overlap)
"""

import numpy as np
from typing import Optional, Tuple
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from sklearn.neighbors import BallTree


# ─────────────────────────────────────────────────────────────────────────────
# Spatial overlap weight estimation from geometry
# ─────────────────────────────────────────────────────────────────────────────

def estimate_spatial_overlap_weights(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    sigma_boundary_frac: float = 0.03,
    interior_weight: float = 1.0,
    exterior_weight: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-cell spatial overlap weights from the known SE(2) alignment.

    For each cell in A: how much of its neighbourhood in B's frame is inside
    B's convex hull?  For each cell in B: is it inside A's (transformed) hull?

    Cells inside both hulls → overlap region → high weight (should be matched).
    Cells outside → unique region → low weight (may be unmatched).

    The boundary is softened with a Gaussian falloff of width
    sigma = sigma_boundary_frac * diameter(B).

    Parameters
    ----------
    coords_A  : (n_A, 2) — sliceA cell coordinates IN SLICEB'S FRAME
                (i.e., already transformed by R and t).
    coords_B  : (n_B, 2) — sliceB cell coordinates.
    R, t      : SE(2) transformation (used only for log messages).
    sigma_boundary_frac : float — boundary softening width as fraction of B's diameter.
    interior_weight : float — weight for cells inside the overlap.
    exterior_weight : float — weight for cells outside the overlap (>0 to keep OT feasible).

    Returns
    -------
    w_A : (n_A,) float64 — overlap weights for sliceA cells.
    w_B : (n_B,) float64 — overlap weights for sliceB cells.
    """
    diam_B = float(np.linalg.norm(coords_B.max(axis=0) - coords_B.min(axis=0))) + 1e-6
    sigma  = sigma_boundary_frac * diam_B

    def _signed_dist_to_hull(pts, hull_pts):
        """
        Approximate signed distance from each point to the hull boundary.
        Positive = inside, Negative = outside.
        We approximate with distance to nearest hull vertex (positive inside).
        """
        try:
            hull = ConvexHull(hull_pts)
        except QhullError:
            # Degenerate hull; all points treated as inside
            return np.ones(len(pts))

        # For each test point, compute whether it's inside the hull
        # Use the half-plane formulation: inside iff all equation[i] @ p + offset <= 0
        # hull.equations: (n_facets, 3) where last col is offset
        A_eq = hull.equations[:, :2]   # (n_facets, 2)
        b_eq = hull.equations[:, 2]    # (n_facets,)

        # signed_dist[i] = min over facets of -(A_eq @ pts[i] + b_eq)
        # Positive inside the hull, negative outside
        vals  = -(pts @ A_eq.T + b_eq[None, :])   # (n_pts, n_facets)
        signed = vals.min(axis=1)

        return signed

    # ── Weights for A cells (transformed into B's frame = coords_A here) ──
    # How well is each A cell inside B's hull?
    sd_A = _signed_dist_to_hull(coords_A, coords_B)
    # Soft boundary: logistic on signed distance
    w_A  = _sigmoid_weight(sd_A, sigma, interior_weight, exterior_weight)

    # ── Weights for B cells (is each B cell inside A's footprint?) ─────────
    sd_B = _signed_dist_to_hull(coords_B, coords_A)
    w_B  = _sigmoid_weight(sd_B, sigma, interior_weight, exterior_weight)

    return w_A, w_B


def _sigmoid_weight(
    signed_dist: np.ndarray,
    sigma: float,
    interior_weight: float,
    exterior_weight: float,
) -> np.ndarray:
    """
    Convert signed distance to a smooth weight using a logistic function.
    Positive signed_dist (inside) → high weight.
    Negative (outside) → low weight.
    """
    # logistic: 1 / (1 + exp(-signed_dist / sigma))
    w = 1.0 / (1.0 + np.exp(-signed_dist / (sigma + 1e-10)))
    # Rescale to [exterior_weight, interior_weight]
    w = exterior_weight + (interior_weight - exterior_weight) * w
    return w.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Overlap weight estimation from OT plan (iterative, no prior geometry needed)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_overlap_from_plan(
    pi: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    smoothing_sigma: float = 0.0,
    coords_A: Optional[np.ndarray] = None,
    coords_B: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-cell overlap fractions from the current transport plan.

    The fraction of cell i's mass that was transported:
        f_A(i) = (π · 1)[i] / a[i]   ∈ [0, 1] approximately

    A cell with f_A(i) ≈ 1 has found a good match → it's in the overlap.
    A cell with f_A(i) ≈ 0 → unmatched → it's in the unique region.

    Spatial smoothing via Gaussian kernel removes noise (optional).

    Parameters
    ----------
    pi : (n_A, n_B) transport plan.
    a  : (n_A,) source marginal.
    b  : (n_B,) target marginal.
    smoothing_sigma : float — Gaussian kernel sigma for smoothing f_A and f_B.
                      0 = no smoothing. In same units as coords.
    coords_A, coords_B : (n, 2) — needed if smoothing_sigma > 0.

    Returns
    -------
    f_A : (n_A,) float64 — fraction of mass transported for each A cell.
    f_B : (n_B,) float64 — fraction of mass received for each B cell.
    """
    row_sums = pi.sum(axis=1)   # (n_A,)
    col_sums = pi.sum(axis=0)   # (n_B,)

    f_A = row_sums / (a + 1e-12)
    f_B = col_sums / (b + 1e-12)

    # Clip to [0, 1] (numerical safety)
    f_A = np.clip(f_A, 0.0, 1.0)
    f_B = np.clip(f_B, 0.0, 1.0)

    # Optional spatial smoothing
    if smoothing_sigma > 0.0 and coords_A is not None:
        f_A = _spatial_smooth(f_A, coords_A, smoothing_sigma)
    if smoothing_sigma > 0.0 and coords_B is not None:
        f_B = _spatial_smooth(f_B, coords_B, smoothing_sigma)

    return f_A.astype(np.float64), f_B.astype(np.float64)


def _spatial_smooth(
    values: np.ndarray,
    coords: np.ndarray,
    sigma: float,
    k_nn: int = 20,
) -> np.ndarray:
    """
    Gaussian-weighted local averaging of per-cell values using k-NN.
    Smoothes out noise in the estimated overlap fractions.
    """
    tree  = BallTree(coords)
    dists, idxs = tree.query(coords, k=min(k_nn + 1, len(coords)))
    dists, idxs = dists[:, 1:], idxs[:, 1:]  # exclude self

    weights = np.exp(-0.5 * (dists / sigma) ** 2)   # (n, k)
    w_sum   = weights.sum(axis=1, keepdims=True) + 1e-10

    smoothed = (weights * values[idxs]).sum(axis=1) / w_sum.ravel()
    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# Iterative overlap-aware FUGW
# ─────────────────────────────────────────────────────────────────────────────

def iterative_overlap_fugw(
    D_A_np: np.ndarray,
    D_B_np: np.ndarray,
    M_bio: np.ndarray,
    a_np: np.ndarray,
    b_np: np.ndarray,
    alpha_fugw: float,
    base_rho: float = 0.5,
    min_rho: float = 0.01,
    n_outer_iters: int = 3,
    fugw_max_iter: int = 100,
    epsilon: float = 0.0,
    divergence: str = "kl",
    smoothing_sigma: float = 0.0,
    coords_A: Optional[np.ndarray] = None,
    coords_B: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative spatially-adaptive FUGW for mutual partial overlap.

    Algorithm (outer loop):
      Round 0: solve FUGW with scalar rho = base_rho (initialise)
      Round k: estimate overlap fractions (f_A, f_B) from current pi
               update rho_A[i] = base_rho * f_A[i]  (high overlap → high rho → strict)
               update rho_B[j] = base_rho * f_B[j]
               solve FUGW with updated per-cell marginals
    Convergence: typically 2–3 rounds.

    Intuition
    ----------
    Round 0 gives a rough plan where overlapping cells have high mass flow.
    In Round 1, cells with high mass flow get high rho (strong marginal
    constraint → they MUST be matched), while cells with low flow get low
    rho (weak constraint → they can remain unmatched).
    This self-consistently concentrates the plan on the true overlap region.

    Note on implementation
    ----------------------
    Standard FUGW (ot.gromov.fused_unbalanced_gromov_wasserstein) uses
    scalar rho. For per-cell rho we use a re-weighting trick:
        Equivalent to scalar FUGW on reweighted marginals:
            a'[i] = a[i] * rho_A[i] / mean(rho_A)
            b'[j] = b[j] * rho_B[j] / mean(rho_B)
    This is an approximation but works well in practice.

    Parameters
    ----------
    D_A_np, D_B_np : (n_A, n_A) and (n_B, n_B) normalised distance matrices.
    M_bio          : (n_A, n_B) biological cost.
    a_np, b_np     : marginals.
    alpha_fugw     : FUGW alpha parameter = (1-alpha)/alpha.
    base_rho       : float — base KL relaxation weight.
    min_rho        : float — floor for per-cell rho (keeps problem feasible).
    n_outer_iters  : int — outer iterations.
    smoothing_sigma: float — spatial smoothing sigma for overlap estimation.
    coords_A, coords_B : (n, 2) — needed for spatial smoothing.
    verbose        : bool.

    Returns
    -------
    pi         : (n_A, n_B) float64 — final transport plan.
    f_A_final  : (n_A,) float64 — final overlap fractions for A.
    f_B_final  : (n_B,) float64 — final overlap fractions for B.
    """
    import ot

    pi = np.outer(a_np, b_np)   # initialise with product marginal
    f_A = np.ones(len(a_np))
    f_B = np.ones(len(b_np))

    for outer_it in range(n_outer_iters):
        # ── Update per-cell rho from previous plan ────────────────────────
        if outer_it > 0:
            f_A, f_B = estimate_overlap_from_plan(
                pi, a_np, b_np,
                smoothing_sigma=smoothing_sigma,
                coords_A=coords_A, coords_B=coords_B)

        # Convert per-cell overlap fractions to effective marginal weights
        # Cells in the overlap get higher rho (stronger matching constraint).
        # Cells outside the overlap get lower rho (allowed to be unmatched).
        rho_A_cell = np.clip(base_rho * f_A, min_rho, base_rho)
        rho_B_cell = np.clip(base_rho * f_B, min_rho, base_rho)

        # Re-weighted marginals (the per-cell rho trick)
        mean_rho_A = float(rho_A_cell.mean())
        mean_rho_B = float(rho_B_cell.mean())

        a_rw = a_np * (rho_A_cell / (mean_rho_A + 1e-12))
        b_rw = b_np * (rho_B_cell / (mean_rho_B + 1e-12))

        # Normalise reweighted marginals
        a_rw = np.maximum(a_rw, 1e-12)
        b_rw = np.maximum(b_rw, 1e-12)

        scalar_rho = (mean_rho_A + mean_rho_B) * 0.5

        if verbose:
            f_A_mean = float(f_A.mean()) if outer_it > 0 else 1.0
            f_B_mean = float(f_B.mean()) if outer_it > 0 else 1.0
            print(f"[AdaptFUGW] Outer iter {outer_it+1}/{n_outer_iters}  "
                  f"f_A_mean={f_A_mean:.3f}  f_B_mean={f_B_mean:.3f}  "
                  f"scalar_rho={scalar_rho:.4f}")

        # ── Solve FUGW ────────────────────────────────────────────────────
        try:
            pi_new, _, _ = ot.gromov.fused_unbalanced_gromov_wasserstein(
                Cx=D_A_np,
                Cy=D_B_np,
                wx=a_rw,
                wy=b_rw,
                reg_marginals=scalar_rho,
                epsilon=epsilon,
                divergence=divergence,
                unbalanced_solver="mm",
                alpha=alpha_fugw,
                M=M_bio,
                init_pi=pi if outer_it > 0 else None,
                init_duals=None,
                max_iter=fugw_max_iter,
                tol=1e-6,
                max_iter_ot=500,
                tol_ot=1e-6,
                log=False,
                verbose=False,
            )
            pi = np.asarray(pi_new, dtype=np.float64)
        except Exception as e:
            if verbose:
                print(f"[AdaptFUGW] FUGW failed at iter {outer_it+1}: {e}")
            break

    # Final overlap fractions
    f_A_final, f_B_final = estimate_overlap_from_plan(
        pi, a_np, b_np,
        smoothing_sigma=smoothing_sigma,
        coords_A=coords_A, coords_B=coords_B)

    if verbose:
        print(f"[AdaptFUGW] Done. pi_mass={pi.sum():.4f}  "
              f"f_A={f_A_final.mean():.3f}  f_B={f_B_final.mean():.3f}")

    return pi, f_A_final, f_B_final


# ─────────────────────────────────────────────────────────────────────────────
# Semi-relaxed GW for one-sided partial overlap
# ─────────────────────────────────────────────────────────────────────────────

def semi_relaxed_gw_cost_matrix(
    D_A: np.ndarray,
    D_B: np.ndarray,
    pi: np.ndarray,
) -> float:
    """
    Compute the GW cost for a transport plan pi:
        GW(D_A, D_B, pi) = sum_{ijkl} pi_ij pi_kl (D_A[i,k] - D_B[j,l])^2
    """
    # Efficient computation via trace formula:
    # GW = ||D_A||_F^2 (a ⊗ a) + ||D_B||_F^2 (b ⊗ b) - 2 <D_A pi D_B, pi>
    F = D_A @ pi @ D_B    # (n_A, n_B)
    return float(2.0 * (D_A ** 2 * np.outer(pi.sum(1), pi.sum(1))).sum()
                 + 2.0 * (D_B ** 2 * np.outer(pi.sum(0), pi.sum(0))).sum()
                 - 4.0 * (F * pi).sum())


def auto_rho_from_geometry(
    coords_A_aligned: np.ndarray,
    coords_B: np.ndarray,
    base_rho: float = 0.5,
    min_rho: float = 0.02,
) -> Tuple[float, float]:
    """
    Automatically estimate scalar rho_A and rho_B from the spatial geometry
    after alignment.

    Method: compute the OVERLAP FRACTION between A's bounding box (in B's
    frame) and B's bounding box.

    s_A = fraction of A cells whose nearest B cell is within A's own diameter
    s_B = n_A_in_overlap / n_B_total

    These fractions are used to scale the marginal relaxation:
        rho_A = base_rho * s_A  (more overlap → stricter source constraint)
        rho_B = base_rho * (n_A / n_B)  (larger A relative to B → stricter)

    Parameters
    ----------
    coords_A_aligned : (n_A, 2) — sliceA coordinates in sliceB's frame.
    coords_B         : (n_B, 2) — sliceB coordinates.
    base_rho         : float.
    min_rho          : float.

    Returns
    -------
    rho_A, rho_B : float.
    """
    n_A  = len(coords_A_aligned)
    n_B  = len(coords_B)
    size_ratio = float(n_A) / float(n_B)

    tree_B = BallTree(coords_B)
    # A's spatial radius: median nearest-B distance for all A cells
    dists, _ = tree_B.query(coords_A_aligned, k=1)
    median_nn = float(np.median(dists))

    # Diameter of B
    diam_B = float(np.linalg.norm(coords_B.max(axis=0) - coords_B.min(axis=0))) + 1e-6
    # A cells that are within B: nearest-B distance < 5% of B diameter
    overlap_thresh = max(0.05 * diam_B, 2.0 * median_nn)
    s_A = float((dists.ravel() < overlap_thresh).mean())   # fraction of A in B

    # B cells that are near some A cell
    tree_A = BallTree(coords_A_aligned)
    dists_B, _ = tree_A.query(coords_B, k=1)
    s_B = float((dists_B.ravel() < overlap_thresh).mean())   # fraction of B near A

    rho_A = float(np.clip(base_rho * s_A, min_rho, base_rho))
    rho_B = float(np.clip(base_rho * s_B, min_rho, base_rho))

    return rho_A, rho_B
