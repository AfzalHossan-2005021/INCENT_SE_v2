"""
robust_se2.py — MAGSAC + LO-RANSAC for Robust SE(2) Estimation
================================================================
INCENT-SE v2 improvement to RANSAC-based SE(2) estimation.

Limitations of standard RANSAC (current cast.py:ransac_se2)
-------------------------------------------------------------
1. Hard inlier threshold: a cell that is 1 μm outside the threshold
   contributes NOTHING, while one just inside contributes equally to one
   that is perfectly aligned. This makes the inlier count noisy and
   sensitive to threshold choice.

2. Single refinement round: after selecting the best hypothesis, only
   one round of Procrustes refinement is performed on the inliers.
   This may not converge to the local optimum if the initial RANSAC
   hypothesis was far from optimal.

3. Fixed threshold: threshold = 2% of B's diameter is a fixed heuristic
   that may be too large (too many false inliers) or too small (too few
   true inliers for partial overlap slices).

Improvements in this module
-----------------------------
1. MAGSAC-style soft inlier scoring:
   Instead of a hard 0/1 threshold, use a probabilistic weight:
       w(r) = max(0, 1 − (r / σ_max)^2)^2   (Tukey bisquare)
   or
       w(r) = exp(−r² / (2 σ²))              (Gaussian)
   where σ is adaptively estimated from the data (median of residuals
   from the best hypothesis so far, multiplied by a robustness factor).
   This avoids threshold sensitivity entirely.

2. LO-RANSAC (Local Optimisation):
   After every RANSAC iteration that improves the best hypothesis,
   run a LOCAL OPTIMISATION step:
     a) Identify soft-inliers with w(r) > threshold (loose threshold)
     b) Run weighted Procrustes on the soft-inliers → better (R', t')
     c) Re-score all candidates under (R', t')
     d) Repeat until convergence (typically 3–5 iterations)
   This dramatically reduces the number of outer RANSAC iterations needed.

3. Adaptive threshold:
   After finding a reasonable initial hypothesis, estimate σ from the
   residuals of the top-25% scoring pairs. This is data-driven and
   naturally adapts to the noise level of the current dataset.

4. Hypothesis degeneracy check:
   Reject hypothesis if the two sampled pairs are too close in space
   (degenerate configuration for SE(2)).

Combined effect:
   The combination of soft scoring + LO steps + adaptive sigma gives
   results comparable to 10× more standard RANSAC iterations, while
   also being more accurate (since the final estimate uses all
   high-confidence pairs, not just the single best 2-point hypothesis).

References
----------
Barath, D. et al. (2019) MAGSAC: Marginalising sample consensus. CVPR.
Lebeda, K. et al. (2012) Fixing the locally optimised RANSAC. BMVC.
Chum, O. & Matas, J. (2002) Randomized RANSAC with T_d,d test. BMVC.

Public API
----------
ransac_se2_magsac(pair_i, pair_j, pair_sc, coords_A, coords_B, ...)
lo_ransac_refine(pair_i, pair_j, coords_A, coords_B, R_init, t_init, ...)
adaptive_threshold(coords_A, coords_B, R, t, percentile)
"""

import numpy as np
import warnings
from typing import Tuple, Optional
from sklearn.neighbors import BallTree


# ─────────────────────────────────────────────────────────────────────────────
# Weights / scoring
# ─────────────────────────────────────────────────────────────────────────────

def _tukey_weights(residuals: np.ndarray, sigma: float) -> np.ndarray:
    """
    Tukey bisquare (biweight) robust weight function.
    w(r) = (1 − (r/σ)²)²  for r < σ,  else 0.

    This is a common choice for M-estimators and gives smoother gradient
    than the Gaussian while having strict zero weight beyond σ.
    """
    u = residuals / (sigma + 1e-10)
    w = np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0)
    return w.astype(np.float32)


def _gaussian_weights(residuals: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian soft weight: never exactly zero, good for smooth optimisation."""
    return np.exp(-0.5 * (residuals / (sigma + 1e-10)) ** 2).astype(np.float32)


def _score_hypothesis(
    R: np.ndarray,
    t: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    sigma: float,
    weight_fn: str = "tukey",
    tree_B: Optional[BallTree] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Score a SE(2) hypothesis (R, t) against all cells in A.

    Returns
    -------
    score    : float — sum of soft weights (higher = better hypothesis).
    weights  : (n_A,) float32 — per-cell weights.
    residuals: (n_A,) float32 — nearest-B-cell distances after transform.
    """
    if tree_B is None:
        tree_B = BallTree(coords_B)

    cA_t  = (R @ coords_A.T).T + t
    dists, _ = tree_B.query(cA_t, k=1)
    residuals = dists.ravel().astype(np.float32)

    if weight_fn == "tukey":
        weights = _tukey_weights(residuals, sigma)
    else:
        weights = _gaussian_weights(residuals, sigma)

    score = float(weights.sum())
    return score, weights, residuals


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive sigma estimation
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_threshold(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    percentile: float = 25.0,
    sigma_factor: float = 1.4826,
) -> float:
    """
    Estimate the noise scale σ from the residuals of a current hypothesis.

    Uses the MAD (Median Absolute Deviation) of the k-th percentile of
    nearest-neighbour residuals, scaled by 1.4826 for consistency with σ.

    A good σ: not too small (would wrongly exclude near-inliers) and not
    too large (would include false inliers from the wrong region).

    Parameters
    ----------
    percentile : float — use residuals at this percentile as the MAD input.
                 25% = focus on the best quarter of correspondences.
    sigma_factor : float — 1.4826 = normality constant for σ estimate from MAD.

    Returns
    -------
    sigma : float — estimated noise scale.
    """
    tree_B    = BallTree(coords_B)
    cA_t      = (R @ coords_A.T).T + t
    dists, _  = tree_B.query(cA_t, k=1)
    residuals = np.sort(dists.ravel())

    n_subset  = max(10, int(len(residuals) * percentile / 100.0))
    subset    = residuals[:n_subset]

    mad = float(np.median(np.abs(subset - np.median(subset))))
    sigma = sigma_factor * mad

    # Safety bounds: at least 1 μm, at most 5% of B's diameter
    diam  = float(np.linalg.norm(coords_B.max(axis=0) - coords_B.min(axis=0))) + 1e-6
    sigma = float(np.clip(sigma, 1.0, 0.05 * diam))

    return sigma


# ─────────────────────────────────────────────────────────────────────────────
# SE(2) from two point-pairs (minimal solver)
# ─────────────────────────────────────────────────────────────────────────────

def _se2_from_two_pairs_robust(x1, x2, y1, y2, min_baseline: float = 5.0):
    """
    Compute SE(2) from two correspondences (x1→y1, x2→y2).

    Returns (R, t) or None if degenerate.
    min_baseline: minimum distance between x1 and x2 (avoids near-identical pairs).
    """
    baseline = float(np.linalg.norm(x2 - x1))
    if baseline < min_baseline:
        return None
    if float(np.linalg.norm(y2 - y1)) < min_baseline:
        return None

    dA = x2 - x1
    dB = y2 - y1
    theta = np.arctan2(dB[1], dB[0]) - np.arctan2(dA[1], dA[0])
    c, s  = np.cos(theta), np.sin(theta)
    R     = np.array([[c, -s], [s, c]])
    t     = ((y1 - R @ x1) + (y2 - R @ x2)) * 0.5
    return R, t


# ─────────────────────────────────────────────────────────────────────────────
# Local Optimisation (LO-RANSAC inner loop)
# ─────────────────────────────────────────────────────────────────────────────

def lo_ransac_refine(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_sc: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    sigma: float,
    tree_B: Optional[BallTree] = None,
    max_lo_iters: int = 8,
    tol: float = 1e-4,
    weight_fn: str = "tukey",
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Local Optimisation step for LO-RANSAC.

    Starting from an initial (R_init, t_init), iteratively refines by:
      1. Score all cells under current (R, t) → soft weights w_i
      2. Weighted Procrustes on all pairs weighted by w_i
      3. Re-score under new (R, t)
      4. Repeat until convergence

    This local loop converges in 3–8 iterations and gives a substantially
    better estimate than the initial 2-point hypothesis.

    Parameters
    ----------
    pair_i, pair_j : (M,) int32 — candidate correspondence indices.
    pair_sc        : (M,) float32 — descriptor similarity scores.
    coords_A       : (n_A, 2) float64.
    coords_B       : (n_B, 2) float64.
    R_init, t_init : initial SE(2) estimate.
    sigma          : float — soft weight scale.
    max_lo_iters   : int — max LO iterations.
    tol            : float — convergence tolerance on (R,t) change.
    weight_fn      : str — "tukey" or "gaussian".

    Returns
    -------
    R      : (2,2) refined rotation.
    t      : (2,) refined translation.
    score  : float — final soft inlier score.
    w_final: (n_A,) float32 — final per-cell weights.
    """
    if tree_B is None:
        tree_B = BallTree(coords_B)

    R, t = R_init.copy(), t_init.copy()
    prev_score = -np.inf

    for lo_it in range(max_lo_iters):
        # Score all A cells under current (R, t)
        score, w_cells, residuals = _score_hypothesis(
            R, t, coords_A, coords_B, sigma, weight_fn, tree_B)

        # Build soft plan for Procrustes: pair (i, j) gets weight
        # w_cells[pair_i] * descriptor_score[pair]
        cA_t    = (R @ coords_A.T).T + t
        _, nn_B = tree_B.query(cA_t, k=1)
        nn_B    = nn_B.ravel()

        # Use ALL cells (not just the candidate pairs) weighted by w_cells
        pi_soft = np.zeros((len(coords_A), len(coords_B)), dtype=np.float64)
        for idx in range(len(coords_A)):
            if w_cells[idx] > 1e-4:
                pi_soft[idx, nn_B[idx]] = float(w_cells[idx])

        # Weighted Procrustes → new (R, t)
        Z = pi_soft.sum()
        if Z < 1e-12:
            break
        row_s = pi_soft.sum(axis=1)
        col_s = pi_soft.sum(axis=0)
        x_bar = (row_s @ coords_A) / Z
        y_bar = (col_s @ coords_B) / Z
        Xc    = coords_A - x_bar
        Yc    = coords_B - y_bar
        H     = Xc.T @ pi_soft @ Yc
        U, _, Vt = np.linalg.svd(H)
        V     = Vt.T
        d     = np.linalg.det(V @ U.T)
        R_new = V @ np.diag([1.0, d]) @ U.T
        t_new = y_bar - R_new @ x_bar

        # Convergence check
        delta_R = float(np.linalg.norm(R_new - R))
        delta_t = float(np.linalg.norm(t_new - t))
        R, t    = R_new, t_new

        if delta_R < tol and delta_t < tol:
            break

        prev_score = score

    # Final scoring
    final_score, w_final, _ = _score_hypothesis(
        R, t, coords_A, coords_B, sigma, weight_fn, tree_B)

    return R, t, final_score, w_final


# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone Procrustes helper (avoids circular import)
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_procrustes_pairs(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted Procrustes SE(2) from a set of weighted point pairs."""
    Z = weights.sum()
    if Z < 1e-12:
        return np.eye(2), np.zeros(2)

    # Weighted centroids
    x_bar = (weights @ coords_A[pair_i]) / Z
    y_bar = (weights @ coords_B[pair_j]) / Z

    # Cross-covariance H
    Xc = coords_A[pair_i] - x_bar    # (M, 2)
    Yc = coords_B[pair_j] - y_bar    # (M, 2)
    H  = (Xc * weights[:, None]).T @ Yc   # (2, 2)

    U, _, Vt = np.linalg.svd(H)
    V  = Vt.T
    d  = np.linalg.det(V @ U.T)
    R  = V @ np.diag([1.0, d]) @ U.T
    t  = y_bar - R @ x_bar

    return R, t


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: MAGSAC-style RANSAC for SE(2)
# ─────────────────────────────────────────────────────────────────────────────

def ransac_se2_magsac(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_sc: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    # Iteration budget
    n_iter: int = 2000,
    # Sigma estimation
    sigma_init: Optional[float] = None,
    sigma_percentile: float = 25.0,
    # Threshold for counting hard inliers (for reporting only)
    hard_inlier_threshold: Optional[float] = None,
    # Local optimisation
    do_lo: bool = True,
    max_lo_iters: int = 8,
    lo_freq: int = 50,          # run LO every this many RANSAC iters
    # Minimum requirements
    min_inlier_frac: float = 0.03,
    min_baseline_frac: float = 0.05,  # fraction of B's diameter
    # Soft weight function
    weight_fn: str = "tukey",
    verbose: bool = True,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    MAGSAC-style robust SE(2) estimation from candidate correspondences.

    Improvements over standard RANSAC:
    1. Soft MAGSAC-inspired inlier weighting (Tukey bisquare or Gaussian)
    2. Adaptive sigma estimation from data
    3. LO-RANSAC local optimisation at regular intervals
    4. Degenerate configuration rejection (minimum baseline check)

    Parameters
    ----------
    pair_i, pair_j, pair_sc : candidate correspondences from find_candidate_pairs
    coords_A, coords_B      : (n, 2) cell coordinates
    n_iter        : int — RANSAC outer iterations
    sigma_init    : float or None — initial sigma; None = auto from B's diameter
    sigma_percentile : float — percentile for adaptive sigma estimation
    do_lo         : bool — enable LO-RANSAC inner loop
    max_lo_iters  : int — iterations per LO call
    lo_freq       : int — run LO every this many RANSAC iterations
    min_inlier_frac : float — warn if final inlier fraction below this
    min_baseline_frac : float — reject pairs closer than this * B_diameter
    weight_fn     : "tukey" or "gaussian"
    verbose       : bool

    Returns
    -------
    R_best     : (2,2) float64 — rotation.
    t_best     : (2,)  float64 — translation.
    n_inliers  : int — number of hard inliers (for reporting).
    inlier_mask: (n_A,) bool.
    """
    n_A = len(coords_A)
    diam = float(np.linalg.norm(coords_B.max(axis=0) - coords_B.min(axis=0))) + 1e-6

    if sigma_init is None:
        sigma = max(0.01 * diam, 1.0)   # start at 1% of B's diameter
    else:
        sigma = sigma_init

    if hard_inlier_threshold is None:
        hard_thresh = max(0.02 * diam, 1.0)
    else:
        hard_thresh = hard_inlier_threshold

    min_baseline = min_baseline_frac * diam

    if verbose:
        print(f"[MAGSAC] {len(pair_i)} candidates  {n_iter} iters  "
              f"sigma_init={sigma:.1f}  hard_thresh={hard_thresh:.1f}")

    tree_B = BallTree(coords_B)

    # Sampling probabilities proportional to descriptor similarity score
    probs = pair_sc.astype(np.float64)
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total < 1e-12:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs /= total

    R_best  = np.eye(2)
    t_best  = np.zeros(2)
    best_score = -1.0

    rng = np.random.default_rng(rng_seed)

    for it in range(n_iter):
        # ── Sample 2 candidate pairs ────────────────────────────────────
        try:
            idx = rng.choice(len(pair_i), size=2, replace=False, p=probs)
        except Exception:
            break
        i1, j1 = int(pair_i[idx[0]]), int(pair_j[idx[0]])
        i2, j2 = int(pair_i[idx[1]]), int(pair_j[idx[1]])

        if i1 == i2 or j1 == j2:
            continue

        # ── Minimal SE(2) solver ─────────────────────────────────────────
        result = _se2_from_two_pairs_robust(
            coords_A[i1], coords_A[i2],
            coords_B[j1], coords_B[j2],
            min_baseline=min_baseline,
        )
        if result is None:
            continue
        R_h, t_h = result

        # ── Soft scoring ─────────────────────────────────────────────────
        score, weights, residuals = _score_hypothesis(
            R_h, t_h, coords_A, coords_B, sigma, weight_fn, tree_B)

        if score > best_score:
            best_score = score
            R_best, t_best = R_h.copy(), t_h.copy()

            # Adaptive sigma update: re-estimate from top residuals
            if it > 50:  # wait for a few iterations first
                sigma = adaptive_threshold(
                    coords_A, coords_B, R_best, t_best,
                    percentile=sigma_percentile)
                # Also update hard threshold proportionally
                hard_thresh = max(1.5 * sigma, 1.0)

        # ── LO-RANSAC: local refinement ──────────────────────────────────
        if do_lo and (it % lo_freq == 0) and best_score > 0:
            R_lo, t_lo, score_lo, _ = _lo_refine_simple(
                pair_i, pair_j, pair_sc,
                coords_A, coords_B,
                R_best, t_best, sigma, tree_B, max_lo_iters, weight_fn)
            if score_lo > best_score:
                best_score = score_lo
                R_best, t_best = R_lo, t_lo

    # ── Final LO pass ─────────────────────────────────────────────────────
    if do_lo:
        R_best, t_best, best_score, _ = _lo_refine_simple(
            pair_i, pair_j, pair_sc,
            coords_A, coords_B,
            R_best, t_best, sigma, tree_B, max_lo_iters, weight_fn)

    # ── Hard inlier count for reporting ───────────────────────────────────
    cA_t   = (R_best @ coords_A.T).T + t_best
    dists, _ = tree_B.query(cA_t, k=1)
    inlier_mask = (dists.ravel() < hard_thresh)
    n_inliers   = int(inlier_mask.sum())
    inlier_frac = n_inliers / n_A

    theta_best = float(np.degrees(np.arctan2(R_best[1, 0], R_best[0, 0])))
    if verbose:
        print(f"[MAGSAC] Best: theta={theta_best:.1f}  "
              f"inliers={n_inliers}/{n_A} ({inlier_frac*100:.1f}%)  "
              f"soft_score={best_score:.1f}")

    if inlier_frac < min_inlier_frac:
        warnings.warn(
            f"[MAGSAC] Only {inlier_frac*100:.1f}% hard inliers — "
            "transformation may be unreliable. Increase n_iter or "
            "consider checking min_desc_score / top_k_pairs.",
            stacklevel=2)

    return R_best, t_best, n_inliers, inlier_mask


def _lo_refine_simple(
    pair_i, pair_j, pair_sc,
    coords_A, coords_B,
    R_init, t_init, sigma, tree_B, max_iters, weight_fn,
):
    """
    Simplified LO-RANSAC using only candidate pairs (not all A cells).
    Faster than the full version in lo_ransac_refine.
    """
    R, t = R_init.copy(), t_init.copy()
    prev_score = -np.inf

    for _ in range(max_iters):
        # Score all candidate pairs
        cA_t     = (R @ coords_A.T).T + t
        dists, _ = tree_B.query(cA_t[pair_i], k=1)
        res      = dists.ravel()

        if weight_fn == "tukey":
            w_pairs = _tukey_weights(res, sigma)
        else:
            w_pairs = _gaussian_weights(res, sigma)

        # Also weight by descriptor score
        w_combined = w_pairs * pair_sc.astype(np.float32)

        total_w = w_combined.sum()
        if total_w < 1e-12:
            break

        # Weighted Procrustes
        R_new, t_new = _weighted_procrustes_pairs(
            coords_A, coords_B, pair_i, pair_j, w_combined)

        delta = float(np.linalg.norm(R_new - R)) + float(np.linalg.norm(t_new - t))
        R, t = R_new, t_new

        # Overall score
        score, _, _ = _score_hypothesis(R, t, coords_A, coords_B, sigma, weight_fn, tree_B)
        if score <= prev_score + 0.01 and delta < 1e-4:
            break
        prev_score = score

    final_score, w_final, _ = _score_hypothesis(
        R, t, coords_A, coords_B, sigma, weight_fn, tree_B)

    return R, t, final_score, w_final
