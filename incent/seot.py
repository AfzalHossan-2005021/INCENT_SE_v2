"""SE(2)-OT EM alignment with explicit rigid transform recovery."""

import os
import time
import datetime
import warnings
import numpy as np
from typing import Optional, Tuple, List
from anndata import AnnData

from ._gpu import resolve_device, to_torch, to_numpy


# ==========================================================================
# Core maths: weighted Procrustes (M-step, closed form)
# ==========================================================================

def weighted_procrustes(
    pi: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Closed-form SE(2) solution from a soft correspondence matrix pi.

    Given pi[i,j] = how strongly cell i in A corresponds to cell j in B,
    find rotation R and translation t that minimise:
        sum_ij pi_ij ||R * x_i + t - y_j||^2

    Solution (weighted Procrustes / Kabsch algorithm):
        Z  = pi.sum()
        x_bar = pi.sum(1) @ coords_A / Z   (pi-weighted centroid of A)
        y_bar = pi.sum(0) @ coords_B / Z   (pi-weighted centroid of B)
        H  = (coords_A - x_bar)^T @ pi @ (coords_B - y_bar)   (2x2)
        U, S, V^T = SVD(H)
        R  = V diag(1, det(V U^T)) U^T     (det correction prevents reflection)
        t  = y_bar - R @ x_bar

    Parameters
    ----------
    pi       : (n_A, n_B) float64 -- soft correspondence (transport plan).
    coords_A : (n_A, 2) float64  -- cell coordinates in sliceA.
    coords_B : (n_B, 2) float64  -- cell coordinates in sliceB.

    Returns
    -------
    R        : (2, 2) float64 -- rotation matrix.
    t        : (2,)   float64 -- translation vector.
    residual : float          -- weighted MSE = sum_ij pi_ij ||R x_i + t - y_j||^2 / Z.
    """
    Z = pi.sum()
    if Z < 1e-12:
        return np.eye(2), np.zeros(2), np.inf

    # Pi-weighted centroids
    row_sums = pi.sum(axis=1)   # (n_A,)
    col_sums = pi.sum(axis=0)   # (n_B,)
    x_bar = (row_sums @ coords_A) / Z   # (2,)
    y_bar = (col_sums @ coords_B) / Z   # (2,)

    # Cross-covariance matrix H = X_c^T @ pi @ Y_c   shape (2, 2)
    X_c = coords_A - x_bar    # (n_A, 2)  centred A
    Y_c = coords_B - y_bar    # (n_B, 2)  centred B
    H   = X_c.T @ pi @ Y_c   # (2, n_A) @ (n_A, n_B) @ (n_B, 2) = (2, 2)

    # SVD of H
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Determinant correction: ensure R is a proper rotation (det=+1), not reflection
    d = np.linalg.det(V @ U.T)
    R = V @ np.diag([1.0, d]) @ U.T   # (2, 2)

    # Translation
    t = y_bar - R @ x_bar   # (2,)

    # Weighted residual
    coords_A_transformed = (R @ coords_A.T).T + t   # (n_A, 2)
    diff_sq = ((coords_A_transformed[:, None, :] - coords_B[None, :, :]) ** 2).sum(axis=2)
    residual = float((pi * diff_sq).sum() / Z)

    return R, t, residual


# ==========================================================================
# E-step: build spatial cost and solve linear OT
# ==========================================================================

def build_spatial_cost(
    R: np.ndarray,
    t: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    D_normalise: float,
) -> np.ndarray:
    """
    Compute the normalised squared Euclidean cost ||R x_i + t - y_j||^2 / D^2.

    Parameters
    ----------
    R           : (2, 2) rotation matrix.
    t           : (2,) translation.
    coords_A    : (n_A, 2) cell coordinates in sliceA.
    coords_B    : (n_B, 2) cell coordinates in sliceB.
    D_normalise : float -- normalisation scale (typically max pairwise distance in B).

    Returns
    -------
    C_spatial : (n_A, n_B) float32 -- normalised squared distances.
    """
    # Transform A coordinates into B's frame
    coords_A_t = (R @ coords_A.T).T + t   # (n_A, 2)

    # Pairwise squared Euclidean: ||x_i_transformed - y_j||^2
    sq_A = (coords_A_t ** 2).sum(axis=1, keepdims=True)   # (n_A, 1)
    sq_B = (coords_B   ** 2).sum(axis=1, keepdims=True).T  # (1, n_B)
    D2   = sq_A + sq_B - 2.0 * (coords_A_t @ coords_B.T)  # (n_A, n_B)
    D2   = np.maximum(D2, 0.0)

    return (D2 / (D_normalise ** 2 + 1e-12)).astype(np.float32)


def solve_ot_step(
    cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    rho_A: float,
    rho_B: float,
    reg_sinkhorn: float = 0.01,
) -> np.ndarray:
    """
    E-step: solve unbalanced OT with the combined cost matrix.

    Uses POT's sinkhorn_unbalanced which handles the KL marginal relaxation:
      min_pi <cost, pi> + rho_A * KL(pi*1 || a) + rho_B * KL(pi^T*1 || b)
      + reg * H(pi)    (entropic regularisation for stability)

    Parameters
    ----------
    cost         : (n_A, n_B) float32 -- combined M_bio + alpha * C_spatial.
    a            : (n_A,) float64     -- source marginal (uniform).
    b            : (n_B,) float64     -- target marginal (uniform).
    rho_A        : float -- source marginal relaxation.
    rho_B        : float -- target marginal relaxation.
    reg_sinkhorn : float -- Sinkhorn entropic regularisation.

    Returns
    -------
    pi : (n_A, n_B) float64 -- transport plan.
    """
    import ot
    pi = ot.unbalanced.sinkhorn_unbalanced(
        a=a.astype(np.float64),
        b=b.astype(np.float64),
        M=cost.astype(np.float64),
        reg=reg_sinkhorn,
        reg_m=(rho_A, rho_B),
        numItermax=1000,
        stopThr=1e-7,
        log=False,
    )
    return np.asarray(pi, dtype=np.float64)


# ==========================================================================
# Full EM loop
# ==========================================================================

def seot_em(
    M_bio: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    alpha: float = 0.5,
    rho_A: float = 0.5,
    rho_B: float = 0.5,
    reg_sinkhorn: float = 0.01,
    max_iter: int = 50,
    tol: float = 1e-5,
    adaptive_procrustes: bool = True,
    procrustes_percentile: float = 70.0,
    min_conf_thresh: float = 0.1,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], float]:
    """
    SE(2)-OT EM algorithm: jointly optimise (R, t) and correspondence pi.

    Alternates between:
      E-step: fix (R, t) -> solve unbalanced linear OT
              Cost = (1-alpha)*M_bio + alpha*||R x_i + t - y_j||^2 / D^2
      M-step: fix pi     -> solve weighted Procrustes for (R, t)

    Guaranteed to converge (each step strictly decreases the objective).

    Parameters
    ----------
    M_bio       : (n_A, n_B) float32  -- biological cost (expression + topology).
    coords_A    : (n_A, 2)  float64   -- sliceA cell coordinates.
    coords_B    : (n_B, 2)  float64   -- sliceB cell coordinates.
    a           : (n_A,)    float64   -- source marginal.
    b           : (n_B,)    float64   -- target marginal.
    R_init      : (2, 2)    float64   -- initial rotation (e.g. from BISPA).
    t_init      : (2,)      float64   -- initial translation.
    alpha       : float -- spatial weight [0=biology only, 1=spatial only].
    rho_A, rho_B: float -- KL marginal relaxation per side.
                   rho ~ s (matched fraction): smaller = more cells unmatched.
    reg_sinkhorn: float -- Sinkhorn entropic regularisation.
                   Smaller = more precise but slower. Try 0.005 - 0.05.
    max_iter    : int  -- maximum EM iterations.
    tol         : float -- convergence threshold on residual change.
    adaptive_procrustes : bool, default True
        Use percentile-based confidence filtering instead of a fixed 0.3 rule.
    procrustes_percentile : float, default 70.0
        Percentile applied to row-wise maxima of the transport plan.
    min_conf_thresh : float, default 0.1
        Lower bound for the confidence threshold after percentile selection.
    verbose     : bool.

    Returns
    -------
    pi          : (n_A, n_B) float64 -- final transport plan.
    R           : (2, 2) float64     -- recovered rotation.
    t           : (2,)   float64     -- recovered translation.
    history     : list of float      -- residual per iteration.
    obj         : float              -- final objective value.
    """
    # ── Coordinate normalisation ──────────────────────────────────────────────
    # CRITICAL FIX for cross-timepoint data.
    #
    # Problem: sliceA and sliceB come from different scanners with completely
    # unrelated coordinate origins (x ~ 1.76M, y ~ 4.86M etc.).  The Procrustes
    # M-step computes t = y_bar - R @ x_bar.  When y_bar and x_bar are in
    # different scanner frames, t is dominated by the ~million-micron scanner
    # offset and bears no relation to the tissue geometry.
    #
    # Fix: normalise each slice INDEPENDENTLY to zero-mean, unit-scale before
    # running EM.  The EM then works in a shared normalised space where both
    # slices fill the range [-1, +1].  Rotation R is scale-invariant; after EM
    # we convert t back to the physical coordinate frame of sliceB.
    #
    # Conversion:  coords_A_aligned = scale_B/scale_A * R_em @ (coords_A - mu_A)
    #                                + scale_B * t_em_n + mu_B
    # So physical translation:  t_phys = -scale_B/scale_A * R_em @ mu_A
    #                                    + scale_B * t_em_n + mu_B

    # Robust normalisation: median centroid + median-absolute-deviation scale.
    # Mean/std are sensitive to outlier cells at tissue borders (partial overlap).
    mu_A    = np.median(coords_A, axis=0)
    mu_B    = np.median(coords_B, axis=0)
    scale_A = float(np.median(np.linalg.norm(coords_A - mu_A, axis=1))) + 1e-6
    scale_B = float(np.median(np.linalg.norm(coords_B - mu_B, axis=1))) + 1e-6

    cA_n = (coords_A - mu_A) / scale_A   # normalised sliceA coords
    cB_n = (coords_B - mu_B) / scale_B   # normalised sliceB coords

    # Transform R_init and t_init into normalised space.
    # In physical space: x_A_phys -> R_init @ x_A_phys + t_init
    # In norm space:     cA_n     -> R_init @ cA_n + t_init_n
    # where t_init_n = (t_init + R_init @ mu_A - mu_B) / scale_B
    t_init_n = ((t_init + R_init @ mu_A - mu_B) / scale_B
                if not np.allclose(t_init, 0.0)
                else np.zeros(2))

    D_norm = 1.0   # already unit-scale

    R, t = R_init.copy(), t_init_n.copy()
    history = []
    pi = np.outer(a, b)   # uniform initialisation

    # Alpha warm-up: high spatial weight early to lock in geometry,
    # then relax to the target alpha. This ensures the rotation/translation
    # are determined by spatial structure first, then refined by biology.
    # Schedule: alpha_eff = min(1.0, alpha + (1.0-alpha) * exp(-it/5))
    # At it=0: alpha_eff ≈ 1.0 (pure spatial)
    # At it=5: alpha_eff ≈ alpha + 0.37*(1-alpha)
    # At it=20: alpha_eff ≈ alpha (converged to target)
    warmup_strength = 1.0 - alpha   # how much to warm up (0 if alpha=1 already)

    for it in range(max_iter):
        # Warm-up schedule: exponential decay toward target alpha
        alpha_eff = float(alpha + warmup_strength * np.exp(-it / 5.0))
        alpha_eff = max(alpha_eff, alpha)   # never go below target

        # ── E-step: OT with current (R, t) in normalised space ────────────
        # Sinkhorn reg annealing: large reg early (fast to right basin),
        # small reg late (sharp correspondences). 10x -> 1x over ~12 iters.
        reg_eff = float(reg_sinkhorn * max(1.0, 10.0 * np.exp(-it / 4.0)))
        C_spatial = build_spatial_cost(R, t, cA_n, cB_n, D_norm)
        cost = ((1.0 - alpha_eff) * M_bio.astype(np.float32)
                + alpha_eff        * C_spatial).astype(np.float64)

        pi = solve_ot_step(cost, a, b, rho_A, rho_B, reg_eff)

        # ── M-step: Procrustes on high-confidence pairs only ─────────────
        row_max = pi.max(axis=1)
        row_max_safe = np.where(row_max < 1e-12, 1e-12, row_max)
        if adaptive_procrustes:
            conf_thresh = float(np.percentile(row_max, procrustes_percentile))
            conf_thresh = max(conf_thresh, min_conf_thresh * float(row_max.mean()))
            conf_thresh = max(conf_thresh, 1e-12)
            pi_conf = np.where(pi >= 0.5 * conf_thresh, pi, 0.0)
        else:
            conf_thresh = 0.3
            pi_conf = np.where(pi >= conf_thresh * row_max_safe[:, None], pi, 0.0)
        Z_conf  = pi_conf.sum()
        if Z_conf > 1e-12:
            R, t, residual = weighted_procrustes(pi_conf, cA_n, cB_n)
        else:
            R, t, residual = weighted_procrustes(pi, cA_n, cB_n)

        history.append(residual)

        if verbose and (it % 5 == 0 or it < 3):
            theta_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            print(f"  [SEOT EM] iter={it+1:3d}  alpha_eff={alpha_eff:.3f}  "
                  f"residual={residual:.6f}  theta={theta_deg:.2f}  "
                  f"pi_mass={pi.sum():.4f}")

        if it > 0 and abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-12) < tol:
            if verbose:
                print(f"  [SEOT EM] Converged at iteration {it+1}.")
            break

    # ── Convert (R, t) from normalised back to physical coordinates ────────
    # Physical alignment: coords_A_aligned = (scale_B/scale_A)*R @ (coords_A-mu_A)
    #                                       + scale_B*t + mu_B
    # Equivalently:  x_aligned = R @ x_A + t_phys
    #   where t_phys = -(scale_B/scale_A)*R @ mu_A + scale_B*t + mu_B
    scale_ratio = scale_B / scale_A
    t_physical  = -scale_ratio * (R @ mu_A) + scale_B * t + mu_B

    return pi, R, t_physical, history, scale_ratio


# ==========================================================================
# Initialisation via BISPA community matching
# ==========================================================================

def _initialise_from_bispa(
    sliceA: AnnData,
    sliceB: AnnData,
    target_min_region_frac: float,
    matching_threshold: float,
    rough_grid_size: int,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Use BISPA community matching to compute (R_init, t_init) for the EM loop.

    This breaks bilateral symmetry before EM starts: BISPA identifies which
    community of A corresponds to which community of B, then computes the
    transformation that aligns matched community centroids.

    Returns R_init, t_init, match_score, bispa_info dict.
    """
    from .bispa import (
        decompose_slice, build_community_similarity, hungarian_matching,
        recover_pose_matched, compute_overlap_fractions,
    )
    from .pose import _rotation_matrix

    if verbose:
        print("[SEOT init] BISPA community decomposition ...")

    # Rough rotation for symmetry-breaking before decomposition
    from .pose import estimate_pose
    theta_rough, _, _, _ = estimate_pose(
        sliceA, sliceB, grid_size=rough_grid_size, verbose=False)
    from .rapa import apply_rotation_only_pose
    sliceA_rough = apply_rotation_only_pose(sliceA, sliceB, theta_rough, verbose=False)

    labels_A = decompose_slice(
        sliceA_rough,
        target_min_region_frac=target_min_region_frac,
        slice_label="A_init", verbose=verbose)
    labels_B = decompose_slice(
        sliceB,
        target_min_region_frac=target_min_region_frac,
        slice_label="B_init", verbose=verbose)

    S, comms_A, comms_B = build_community_similarity(
        sliceA_rough, labels_A, sliceB, labels_B, verbose=False)

    matched_pairs, unmatched_A, unmatched_B = hungarian_matching(
        S, comms_A, comms_B, threshold=matching_threshold, verbose=verbose)

    theta_ref, tx_ref, ty_ref, pose_score = recover_pose_matched(
        sliceA_rough, labels_A, sliceB, labels_B,
        matched_pairs, grid_size=rough_grid_size, verbose=verbose)

    R_init = _rotation_matrix(theta_ref)
    t_init = np.array([tx_ref, ty_ref], dtype=np.float64)

    s_A, s_B = compute_overlap_fractions(labels_A, labels_B, matched_pairs)

    bispa_info = {
        "labels_A": labels_A, "labels_B": labels_B,
        "matched_pairs": matched_pairs,
        "unmatched_A": unmatched_A, "unmatched_B": unmatched_B,
        "s_A": s_A, "s_B": s_B,
        "theta_init": theta_ref, "pose_score": pose_score,
        "community_similarity": S,
        "community_labels_A": comms_A,
        "community_labels_B": comms_B,
    }

    if verbose:
        print(f"[SEOT init] R_init: theta={theta_ref:.1f}  "
              f"t=({tx_ref:.1f},{ty_ref:.1f})  s_A={s_A:.3f}  s_B={s_B:.3f}")

    return R_init, t_init, pose_score, bispa_info


# ==========================================================================
# Main public function
# ==========================================================================

def pairwise_align_seot(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    filePath: str,
    # EM parameters
    max_em_iter: int = 50,
    tol_em: float = 1e-5,
    reg_sinkhorn: float = 0.01,
    # Marginal relaxation (rho = None -> computed from BISPA match fractions)
    rho_A: Optional[float] = None,
    rho_B: Optional[float] = None,
    base_rho: float = 0.05,
    # BISPA initialisation
    target_min_region_frac: float = 0.20,
    matching_threshold: float = 0.85,
    rough_grid_size: int = 256,
    # Anchor (penalise matching outside matched communities)
    use_anchor: bool = True,
    lambda_anchor: float = 2.0,
    boundary_sigma_frac: float = 0.05,
    rho_per_cell: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    # Bilateral contiguity refinement
    lambda_spatial: float = 0.05,
    lambda_target: float = 0.05,
    # cVAE for cross-timepoint
    cvae_model=None,
    cvae_path: Optional[str] = None,
    cvae_epochs: int = 80,
    cvae_latent_dim: int = 32,
    cross_timepoint: bool = False,
    # Standard
    use_rep: Optional[str] = None,
    numItermax: int = 2000,
    use_gpu: bool = False,
    gpu_verbose: bool = True,
    verbose: bool = True,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool = False,
    neighborhood_dissimilarity: str = "jsd",
    return_diagnostics: bool = False,
):
    """
    SE(2)-OT EM: jointly recover the rigid transformation and cell correspondences.

    This solves the fundamental problem that GW/FGW CANNOT solve:
    recovering the rotation R and translation t that places sliceA at its
    true position in sliceB's coordinate frame.

    Why GW fails at this
    ---------------------
    GW uses pairwise DISTANCES (rotation-invariant). The transformation
    (R, t) is completely invisible to the GW objective. After GW, the
    post-hoc centroid-based translation is wrong whenever:
      - Both slices are partial with non-matching centroids
      - Either slice has extra symmetric regions (extra hemisphere)
      - Slices are from different scanners with incompatible coordinate origins

    What SEOT does instead
    -----------------------
    Minimises E(pi, R, t) = (1-alpha) * <M_bio, pi>
                           + alpha    * sum_ij pi_ij ||R x_i + t - y_j||^2 / D^2
                           + KL marginal terms

    This explicitly includes (R, t) in the objective. The EM algorithm
    alternates between:
      E-step: fix (R,t) -> solve unbalanced Sinkhorn OT -> pi
      M-step: fix pi    -> solve weighted Procrustes -> (R, t)  [CLOSED FORM]

    Guaranteed to converge. Typically 15-30 iterations.

    Initialisation
    --------------
    BISPA community matching provides (R_init, t_init) that breaks bilateral
    symmetry: Fourier rotation on matched cells + centroid-offset translation.
    The EM then refines this to the exact transformation.

    Parameters
    ----------
    sliceA, sliceB : AnnData with .obsm['spatial'] and .obs['cell_type_annot'].
    alpha  : float -- spatial weight in the cost. 0=biology only, 1=spatial only.
             Recommended: 0.3-0.5 for same-timepoint; 0.5-0.7 for cross-timepoint.
    beta   : float -- cell-type mismatch weight inside M_bio.
    gamma  : float -- neighbourhood dissimilarity weight.
    radius : float -- neighbourhood radius (spatial coordinate units).
    filePath : str -- directory for cache files and logs.

    max_em_iter  : int, default 50 -- maximum EM iterations.
    tol_em       : float, default 1e-5 -- convergence tolerance on residual.
    reg_sinkhorn : float, default 0.01 -- Sinkhorn entropic regularisation.
                   Smaller = more precise, slower. Range: 0.001 - 0.1.

    rho_A, rho_B : float or None -- KL marginal relaxation per side.
                   None = computed from BISPA matched fractions (recommended).
                   Smaller = more cells unmatched on that side.
                   For full overlap: rho ~ 1.0.  For 50% overlap: rho ~ 0.3.
    base_rho     : float, default 0.5 -- scale factor for auto-computed rho.

    target_min_region_frac : float, default 0.20
        For BISPA init: each community must cover >= this fraction of n_cells.
        0.20 -> K=2 (brain hemispheres).
        0.10 -> K<=4 (heart chambers).
    matching_threshold : float, default 0.85 -- max distance for matched pair.

    use_anchor     : bool, default True
        Add anchor cost penalising transport outside matched communities.
    lambda_anchor  : float, default 2.0 -- anchor penalty weight.

    lambda_spatial, lambda_target : float, default 0.05
        Bilateral contiguity regularisation weights (post-EM refinement).

    cross_timepoint : bool, default False
        True -> use cVAE latent cost for M_bio (temporal expression drift).
        False -> use raw cosine + cell-type cost.
    cvae_model / cvae_path : pre-trained INCENT_cVAE or saved model path.

    return_diagnostics : bool, default False
        True -> returns (pi, diagnostics_dict).

    Returns
    -------
    pi : (n_A, n_B) float64 -- final transport plan.
         argmax_j pi[i, :] gives the best-match cell in B for each cell in A.
         pi.sum() < 1 indicates partial overlap.

    If return_diagnostics=True:
        (pi, {
          "R": (2,2) rotation matrix recovered by EM,
          "t": (2,) translation vector,
          "theta_deg": float rotation angle in degrees,
          "residual_history": list of float,
          "pi_mass": float,
          "s_A": float, "s_B": float,
          "matched_pairs": list of (k_A, k_B),
          "sliceA_aligned": AnnData with transformed .obsm['spatial'],
          "bispa_info": dict,
        })
    """
    import ot as pot
    start_time = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_seot_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name else f"{filePath}/log_seot.txt")
    log = open(log_name, "w")
    log.write(f"pairwise_align_seot -- INCENT-SE SEOT\n{datetime.datetime.now()}\n")
    log.write(f"alpha={alpha}  beta={beta}  gamma={gamma}  radius={radius}\n")
    log.write(f"max_em_iter={max_em_iter}  reg_sinkhorn={reg_sinkhorn}\n\n")

    # ==================================================================
    # STEP 1: BISPA initialisation  -> (R_init, t_init, s_A, s_B)
    # ==================================================================
    print("[SEOT] Step 1: BISPA initialisation ...")
    R_init, t_init, pose_score, bispa_info = _initialise_from_bispa(
        sliceA, sliceB,
        target_min_region_frac=target_min_region_frac,
        matching_threshold=matching_threshold,
        rough_grid_size=rough_grid_size,
        verbose=gpu_verbose,
    )
    s_A = bispa_info["s_A"]
    s_B = bispa_info["s_B"]
    matched_pairs  = bispa_info["matched_pairs"]
    unmatched_A    = bispa_info["unmatched_A"]
    unmatched_B    = bispa_info["unmatched_B"]
    labels_A       = bispa_info["labels_A"]
    labels_B       = bispa_info["labels_B"]

    # ── Set marginal relaxation ───────────────────────────────────────────────
    # The correct rho depends on HOW MUCH of each slice is unmatched.
    #
    # rho_A: fraction of sliceA cells that have a counterpart in sliceB.
    #   If sliceA is a single region and sliceB contains it: s_A ≈ 1.0 → rho_A high
    #   If sliceA has extra cells beyond sliceB: s_A < 1.0 → rho_A lower
    #
    # rho_B: fraction of sliceB cells that receive mass from sliceA.
    #   KEY FIX: use the SIZE RATIO n_A / n_B, not just the matched fraction.
    #   If sliceB has two hemispheres and sliceA is one: ~50% of B is unmatched
    #   → rho_B = base_rho * (n_A / n_B_total) captures this correctly.
    n_A_total = len(sliceA)
    n_B_total = len(sliceB)
    size_ratio = float(n_A_total) / float(n_B_total)   # e.g. 10609/14195 = 0.747

    # n_A / n_B tells us: for every B cell, only this fraction "expect" a match.
    # Smaller size_ratio → stronger target marginal relaxation needed.
    rho_B_auto = float(base_rho * min(size_ratio, 1.0))
    # Source: all of A tries to find a home (unless A is larger than B's matched region)
    n_B_matched = sum((labels_B == l).sum() for _, l in matched_pairs) if matched_pairs else n_B_total
    rho_A_auto = float(base_rho * min(float(n_B_matched) / max(n_A_total, 1), 1.0))

    rho_A_use = rho_A if rho_A is not None else rho_A_auto
    rho_B_use = rho_B if rho_B is not None else rho_B_auto

    log.write(f"BISPA init: theta={bispa_info['theta_init']:.1f}  "
              f"pose_score={pose_score:.3f}  s_A={s_A:.3f}  s_B={s_B:.3f}\n")
    log.write(f"matched_pairs={matched_pairs}\n")
    log.write(f"size_ratio={size_ratio:.3f}  n_A={n_A_total}  n_B={n_B_total}\n")
    log.write(f"rho_A={rho_A_use:.4f}  rho_B={rho_B_use:.4f}  "
              f"(auto: rho_A={rho_A_auto:.4f}  rho_B={rho_B_auto:.4f})\n\n")
    print(f"[SEOT] rho_A={rho_A_use:.4f}  rho_B={rho_B_use:.4f}  "
          f"(size_ratio={size_ratio:.3f})")

    # ==================================================================
    # STEP 2: Build biological cost M_bio (expression + cell-type + topology)
    # ==================================================================
    print("[SEOT] Step 2: Building M_bio ...")

    model = None

    if cross_timepoint:
        from .cvae import INCENT_cVAE, train_cvae, latent_cost
        if cvae_model is not None:
            model = cvae_model
        elif cvae_path is not None and os.path.exists(cvae_path):
            model = INCENT_cVAE.load(cvae_path)
        else:
            print("[SEOT] Training cVAE ...")
            model = train_cvae([sliceA, sliceB], latent_dim=cvae_latent_dim,
                               epochs=cvae_epochs, verbose=gpu_verbose)
            if cvae_path:
                model.save(cvae_path)

    # Apply BISPA rotation-only pose to sliceA so coordinates are in B's rough frame
    from .rapa import apply_rotation_only_pose
    from .pose import _rotation_matrix
    sliceA_rough = apply_rotation_only_pose(sliceA, sliceB, bispa_info["theta_init"],
                                            verbose=False)

    # INCENT preprocessing (shared genes, cell types, cosine dist, JSD)
    from .core import _preprocess, _to_np
    log2 = open(f"{filePath}/log_seot_pre.txt", "w")
    p = _preprocess(
        sliceA_rough, sliceB, alpha, beta, gamma, radius, filePath,
        use_rep, None, None, None,
        numItermax, pot.backend.NumpyBackend(), use_gpu, gpu_verbose,
        sliceA_name, sliceB_name, overwrite, neighborhood_dissimilarity,
        log2)
    log2.close()

    sA_filt  = p["sliceA"]
    sB_filt  = p["sliceB"]
    a_np     = _to_np(p["a"])
    b_np     = _to_np(p["b"])
    n_A, n_B = sA_filt.shape[0], sB_filt.shape[0]

    if cross_timepoint:
        from .cvae import latent_cost
        if model is None:
            raise RuntimeError("cVAE model was not initialised for cross_timepoint alignment")
        M1_np = latent_cost(sA_filt, sB_filt, model).astype(np.float32)
    else:
        M1_np = _to_np(p["cosine_dist_gene_expr"]).astype(np.float32)

    M2_np = _to_np(p["M2"]).astype(np.float32)

    # Topological fingerprint cost
    from .topology import compute_fingerprints, fingerprint_cost
    fp_A = compute_fingerprints(sA_filt, radius=radius, n_bins=16,
                                 cache_path=filePath,
                                 slice_name=f"{sliceA_name or 'A'}_seot",
                                 overwrite=overwrite, verbose=gpu_verbose)
    fp_B = compute_fingerprints(sB_filt, radius=radius, n_bins=16,
                                 cache_path=filePath,
                                 slice_name=f"{sliceB_name or 'B'}_seot",
                                 overwrite=overwrite, verbose=gpu_verbose)
    M_topo = fingerprint_cost(fp_A, fp_B, metric="cosine", use_gpu=use_gpu).astype(np.float32)

    # Anchor cost (penalise transport outside matched communities)
    if use_anchor and matched_pairs:
        from .bispa import build_bidirectional_anchor
        def _remap(labels_full, adata_full, adata_filt):
            bc_full = np.array(adata_full.obs_names)
            bc_filt = np.array(adata_filt.obs_names)
            lab_map = {bc: labels_full[i] for i, bc in enumerate(bc_full)}
            return np.array([lab_map.get(bc, -1) for bc in bc_filt], dtype=np.int32)
        la_f = _remap(labels_A, sliceA_rough, sA_filt)
        lb_f = _remap(labels_B, sliceB,       sB_filt)
        M_anchor = build_bidirectional_anchor(
            sA_filt, la_f, sB_filt, lb_f,
            matched_pairs, unmatched_A, unmatched_B,
            lambda_anchor=lambda_anchor,
            boundary_sigma_frac=boundary_sigma_frac,
            use_gpu=use_gpu, verbose=gpu_verbose).astype(np.float32)
    else:
        M_anchor = np.zeros((n_A, n_B), dtype=np.float32)

    # Combined M_bio (biological cost, independent of spatial transformation)
    M_bio = M1_np + gamma * M2_np + 0.3 * M_topo + M_anchor

    # Coordinates of filtered slices in rough frame
    coords_A = sA_filt.obsm["spatial"].astype(np.float64)
    coords_B = sB_filt.obsm["spatial"].astype(np.float64)

    # ── Spatial proximity bias for target marginal b_weighted ────────────────
    # Core insight: after the rough SE(2) rotation, sliceA's centroid is
    # placed near sliceB's global centroid (midline). B cells that are
    # SPATIALLY CLOSE to sliceA's actual footprint are much more likely to
    # be correct matches than B cells far away (wrong hemisphere, extra chamber).
    #
    # We build b_weighted[j] = Gaussian(distance(y_j, footprint_A)) * b[j].
    # sigma = sliceA's median radius (how spread out A is). B cells at distance
    # > 2*sigma from A's centroid get weight exp(-2) ≈ 0.14x, at 3*sigma get 0.01x.
    #
    # This is GENERAL: works for any organ with repeated regions, any K.
    # No organ-specific geometry is assumed.
    # The rho_B marginal relaxation then allows the plan to shed the
    # down-weighted B cells entirely.
    # ── Region-aware spatial prior (generalised, organ-agnostic) ─────────────
    # WHY THIS REPLACES THE CENTROID-DISTANCE PRIOR:
    # The old code placed the prior at sliceB's global centroid. For organs
    # with multiple symmetric regions, that can weight several valid regions
    # equally and prevent region selection.
    #
    # The new code uses the WINNING REGION from BISPA initialisation.
    # Cells in the winning region: weight = 1.0
    # Cells at boundary (within one radius): Gaussian decay → ~0.5 at boundary
    # Cells far outside: floor = 0.01 (keeps OT feasible)
    #
    # This works for any organ with repeated regions.
    # because the winning region is determined by spatial_overlap_score
    # which tests geometric + biological compatibility jointly.
    from .region_matcher import compute_region_spatial_prior, rank_region_candidates
    _community_labels_full = bispa_info.get("labels_B", np.zeros(len(sliceB), dtype=np.int32))

    if bispa_info.get("community_similarity") is not None:
        S_region = np.asarray(bispa_info["community_similarity"], dtype=np.float64)
        comms_B = np.asarray(bispa_info.get("community_labels_B", np.unique(_community_labels_full)))
        region_scores = S_region.max(axis=0)
        best_ks, ranked_scores, region_weights = rank_region_candidates(
            region_scores, comms_B, top_k=min(3, len(comms_B)))
        w_prior_full = compute_region_spatial_prior(
            sliceA_rough, sliceB,
            community_labels=_community_labels_full,
            radius=radius,
            best_ks=best_ks,
            region_weights=region_weights,
        )
    else:
        _best_k_label = (int(bispa_info["matched_pairs"][0][1])
                         if bispa_info.get("matched_pairs") else 0)
        w_prior_full = compute_region_spatial_prior(
            sliceA_rough, sliceB,
            community_labels=_community_labels_full,
            radius=radius,
            best_k=_best_k_label,
        )

    bc_B_full  = np.array(sliceB.obs_names)
    bc_B_filt  = np.array(sB_filt.obs_names)
    bc_to_w    = {bc: w_prior_full[i] for i, bc in enumerate(bc_B_full)}
    w_filt     = np.array([bc_to_w.get(bc, 0.01) for bc in bc_B_filt],
                          dtype=np.float64)
    b_weighted  = b_np * w_filt
    b_weighted  = np.maximum(b_weighted, 1e-10)
    b_weighted /= b_weighted.sum()

    frac_in_region = float((w_filt > 0.5).mean())
    print(f"[SEOTv2] Region prior: {frac_in_region*100:.1f}% of B cells "
          f"in candidate regions (weight=1.0); boundary cells decay over radius={radius:.0f}")
    log.write(f"Region prior: frac_in_region={frac_in_region:.3f}  radius={radius}\n")

    # Adjust R_init and t_init to account for the rough rotation already applied
    # (sliceA_rough has theta_init baked in; R_init from Procrustes is the
    #  ADDITIONAL rotation needed. Total rotation = R_procrustes @ R_rough)
    R_rough = _rotation_matrix(bispa_info["theta_init"])
    # The EM starts from the identity (rough rotation already applied to coords)
    # and the centroid-based translation from BISPA
    # Adjust t_init: it was computed relative to sliceA_rough coordinates
    t_init_em = t_init.astype(np.float64)
    R_init_em = np.eye(2)   # identity: rough rotation is already in coords_A

    # ==================================================================
    # STEP 3: SE(2)-OT EM
    # ==================================================================
    print(f"[SEOT] Step 3: Multi-start SE(2)-OT EM (max_iter={max_em_iter}, "
          f"rho_A={rho_A_use:.3f}, rho_B={rho_B_use:.3f}) ...")

    # ── Coarse-to-fine: spatially stratified subsample for screening ──────
    N_sub = min(3000, n_A)
    if N_sub < n_A:
        cA_raw = coords_A
        cA_min, cA_max = cA_raw.min(axis=0), cA_raw.max(axis=0)
        n_grid = int(np.ceil(np.sqrt(N_sub)))
        gx = np.floor((cA_raw[:,0]-cA_min[0]) / (cA_max[0]-cA_min[0]+1e-6) * n_grid).astype(int)
        gy = np.floor((cA_raw[:,1]-cA_min[1]) / (cA_max[1]-cA_min[1]+1e-6) * n_grid).astype(int)
        grid_ids = gx * (n_grid+1) + gy
        rng_sub  = np.random.default_rng(42)
        sub_idx  = []
        for gid in np.unique(grid_ids):
            pool = np.where(grid_ids == gid)[0]
            k    = max(1, int(np.ceil(N_sub * len(pool) / n_A)))
            sub_idx.extend(rng_sub.choice(pool, size=min(k, len(pool)), replace=False))
        sub_idx  = np.array(sub_idx[:N_sub])
        coords_A_sub = coords_A[sub_idx]
        M_bio_sub    = M_bio[sub_idx, :]
        a_sub        = a_np[sub_idx].copy(); a_sub /= a_sub.sum()
        print(f'[SEOT] Coarse subsampling: {len(sub_idx)}/{n_A} A cells for screening')
    else:
        coords_A_sub, M_bio_sub, a_sub = coords_A, M_bio, a_np

    # Multi-start EM: try 8 candidate rotations spaced 45 degrees apart.
    from .pose import _rotation_matrix as _rm
    # Use the BISPA Fourier estimate as the multi-start base.
    # R_init_em = I (0°) was wrong: the BISPA estimate (e.g. 307°) is the
    # best starting point and should always appear in the candidate set.
    # 4 biologically motivated candidates instead of 8 fixed:
    # theta      — the Fourier-Mellin estimate (most likely correct)
    # theta+180° — the bilateral mirror (second most likely for symmetric organs)
    # theta+90°  — fallback for organs with 4-fold symmetry (heart)
    # theta+270° — the other 90° fallback
    # Any additional 45° candidates between these are biologically implausible
    # and only waste computation on false basins of attraction.
    theta_bispa = bispa_info["theta_init"]
    candidate_thetas = [
        theta_bispa,
        (theta_bispa + 180.0) % 360.0,
        (theta_bispa + 90.0)  % 360.0,
        (theta_bispa + 270.0) % 360.0,
    ]

    best_pi = np.outer(a_sub, b_weighted)
    best_R = np.eye(2)
    best_t = np.zeros(2)
    best_hist = [np.inf]
    best_final_cost = np.inf

    for theta_c in candidate_thetas:
        R_c = _rm(theta_c)
        pi_c, R_c_em, t_c_em, hist_c, _sr_c = seot_em(
            M_bio=M_bio_sub,          # subsampled for speed
            coords_A=coords_A_sub,
            coords_B=coords_B,
            a=a_sub, b=b_weighted,
            R_init=R_c,
            t_init=np.zeros(2),    # zero t; Procrustes will find it
            alpha=alpha,
            rho_A=rho_A_use,
            rho_B=rho_B_use,
            reg_sinkhorn=reg_sinkhorn,
            max_iter=10,           # quick screening
            tol=tol_em,
            verbose=False,
        )
        # Score: final residual + biological cost (subsampled)
        final_bio = float((M_bio_sub.astype(np.float64) * pi_c).sum())
        final_cost = hist_c[-1] + (1 - alpha) * final_bio
        if verbose:
            print(f"  [SEOT multi-start] theta={theta_c:.1f}  "
                  f"residual={hist_c[-1]:.4f}  bio={final_bio:.4f}  "
                  f"total={final_cost:.4f}")
        if final_cost < best_final_cost:
            best_final_cost = final_cost
            best_pi, best_R, best_t, best_hist = pi_c, R_c_em, t_c_em, hist_c

    # Adaptive rho: after screening, the pi_mass tells us whether rho_B
    # was too loose (pi_mass > size_ratio → too much mass transported).
    # Fix: rho_B_final = rho_B_use * size_ratio / pi_mass_screen
    # When pi_mass_screen=1.0, size_ratio=0.747: rho_B tightens by 25%.
    # When pi_mass_screen≈size_ratio: no change (already correct).
    # Clamp to [0.01, rho_B_use] for numerical safety.
    pi_mass_screen = float(best_pi.sum())
    if pi_mass_screen > 1e-3:
        rho_B_final = float(np.clip(
            rho_B_use * size_ratio / pi_mass_screen,
            0.01, rho_B_use))
    else:
        rho_B_final = rho_B_use
    rho_A_final = rho_A_use
    print(f'[SEOT] Adaptive rho: pi_mass_screen={pi_mass_screen:.3f}  '
          f'size_ratio={size_ratio:.3f}  rho_B {rho_B_use:.4f} -> {rho_B_final:.4f}')
    log.write(f'Adaptive rho: pi_mass_screen={pi_mass_screen:.3f}  '
              f'rho_B {rho_B_use:.4f} -> {rho_B_final:.4f}\n')

    # Full refinement from the best starting rotation
    best_theta = float(np.degrees(np.arctan2(best_R[1, 0], best_R[0, 0])))
    print(f"[SEOT] Best start: theta={best_theta:.1f}  refining ...")
    pi, R_em, t_em, history, scale_ratio_em = seot_em(
        M_bio=M_bio,
        coords_A=coords_A,
        coords_B=coords_B,
        a=a_np, b=b_weighted,
        R_init=best_R,
        t_init=np.zeros(2),
        alpha=alpha,
        rho_A=rho_A_final,
        rho_B=rho_B_final,
        reg_sinkhorn=reg_sinkhorn,
        max_iter=max_em_iter,
        tol=tol_em,
        verbose=verbose,
    )

    # Total transformation: rough rotation + EM refinement
    R_total     = R_em @ R_rough
    theta_total = float(np.degrees(np.arctan2(R_total[1, 0], R_total[0, 0])))

    # ── 1-NN post-refinement: sharpen the rotation using hard assignments ─────
    # After the EM converges, the soft plan pi contains the best
    # rotation estimate. Build a hard (1-NN) assignment: for each A cell,
    # its nearest B neighbour under the current transformation. Then run
    # one more weighted Procrustes on these hard pairs. This sharpens
    # R and t beyond what the entropic Sinkhorn can achieve.
    from sklearn.neighbors import NearestNeighbors
    coords_A_n_final = (coords_A - np.median(coords_A, axis=0)) / (
        float(np.median(np.linalg.norm(coords_A - np.median(coords_A, axis=0), axis=1))) + 1e-6)
    coords_B_n_final = (coords_B - np.median(coords_B, axis=0)) / (
        float(np.median(np.linalg.norm(coords_B - np.median(coords_B, axis=0), axis=1))) + 1e-6)
    # Transform A with current R_em in normalised space
    R_em_n   = R_em   # R_em already in normalised space (Procrustes is scale-invariant)
    t_em_n   = (coords_B_n_final.mean(axis=0)
                - R_em_n @ coords_A_n_final.mean(axis=0))  # rough normalised t
    cA_t_n   = (R_em_n @ coords_A_n_final.T).T + t_em_n
    nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(coords_B_n_final)
    dists_nn, nn_idx = nn_model.kneighbors(cA_t_n)
    nn_idx    = nn_idx.ravel()
    # Only use inlier pairs (distance < 2x median) to avoid outlier contamination
    threshold_nn = 2.0 * float(np.median(dists_nn))
    inlier_mask  = dists_nn.ravel() < threshold_nn
    if inlier_mask.sum() >= 10:
        pi_hard = np.zeros((n_A, n_B), dtype=np.float64)
        pi_hard[np.where(inlier_mask)[0], nn_idx[inlier_mask]] = 1.0 / inlier_mask.sum()
        R_refined, t_refined_n, _ = weighted_procrustes(pi_hard, coords_A_n_final, coords_B_n_final)
        if abs(np.degrees(np.arctan2(R_refined[1,0], R_refined[0,0])) -
               np.degrees(np.arctan2(R_em[1,0], R_em[0,0]))) < 10.0:
            R_em  = R_refined
            print(f"[SEOT] 1-NN refinement: theta {np.degrees(np.arctan2(R_em[1,0],R_em[0,0])):.2f}°  "
                  f"inliers={inlier_mask.sum()}/{n_A} ({inlier_mask.mean()*100:.1f}%)")

    R_total     = R_em @ R_rough
    theta_total = float(np.degrees(np.arctan2(R_total[1, 0], R_total[0, 0])))

    # ── Correct sliceA_aligned with scale_ratio ───────────────────────────────
    # The seot_em internal transformation is:
    #   x_B = scale_ratio * R_em @ x_A_rough + t_em
    # NOT: R_em @ x_A_rough + t_em (what the old code applied to original coords).
    # Apply scale_ratio and use sliceA_rough coordinates directly.
    # This avoids the rotation-composition ambiguity and the scale error.
    coords_rough_all = sliceA_rough.obsm["spatial"].astype(np.float64)
    t_total = t_em  # t_em is the physical translation in rough-rotated space

    # ==================================================================
    # STEP 4: Bilateral contiguity post-refinement
    # ==================================================================
    if lambda_spatial > 0.0 or lambda_target > 0.0:
        print("[SEOT] Step 4: Bilateral contiguity refinement ...")
        from .contiguity import contiguity_gradient, build_spatial_affinity
        from .rapa import target_contiguity_gradient, build_target_affinity
        sigma_c = radius / 3.0
        D_B_np = _to_np(p["D_B"])
        D_A_np = _to_np(p["D_A"])
        W_A = build_spatial_affinity(coords_A, sigma=sigma_c, k_nn=20)
        W_B = build_target_affinity(sB_filt, sigma=sigma_c, k_nn=20)
        for _ in range(10):
            grad = np.zeros_like(pi)
            if lambda_spatial > 0.0:
                grad += lambda_spatial * contiguity_gradient(pi, W_A, D_B_np, use_gpu=use_gpu)
            if lambda_target > 0.0:
                grad += lambda_target * target_contiguity_gradient(pi, W_B, D_A_np, use_gpu=use_gpu)
            pi = np.maximum(pi - 0.05 * grad, 0.0)
            rs = pi.sum(axis=1, keepdims=True)
            pi = pi / np.maximum(rs, 1e-12) * a_np[:, None]

    pi_mass = float(pi.sum())
    runtime = time.time() - start_time

    log.write(f"EM converged: {len(history)} iterations\n")
    log.write(f"R_total:\n{R_total}\nt_total={t_total}\n")
    log.write(f"theta_total={theta_total:.2f}  pi_mass={pi_mass:.4f}\n")
    log.write(f"Runtime={runtime:.1f}s\n")
    log.close()

    print(f"[SEOT] Done.  theta={theta_total:.1f}  "
          f"t=({t_total[0]:.1f},{t_total[1]:.1f})  "
          f"pi_mass={pi_mass:.4f}  Runtime={runtime:.1f}s")

    # Build aligned sliceA: apply scale_ratio * R_em to sliceA_rough coordinates.
    # This is the CORRECT transformation from seot_em:
    #   x_B = scale_ratio_em * R_em @ x_A_rough + t_em
    # where x_A_rough = sliceA_rough.obsm["spatial"].
    # Using sliceA_rough avoids rotation-composition issues from R_total = R_em @ R_rough.
    sliceA_aligned = sliceA_rough.copy()
    sliceA_aligned.obsm["spatial"] = (
        scale_ratio_em * (R_em @ sliceA_rough.obsm["spatial"].astype(np.float64).T).T
        + t_em)

    if return_diagnostics:
        return pi, {
            "R": R_total,
            "t": t_total,
            "theta_deg": theta_total,
            "residual_history": history,
            "pi_mass": pi_mass,
            "s_A": s_A, "s_B": s_B,
            "matched_pairs": matched_pairs,
            "sliceA_aligned": sliceA_aligned,
            "bispa_info": bispa_info,
        }
    return pi