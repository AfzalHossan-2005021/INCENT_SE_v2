"""CAST alignment pipeline for partial-overlap spatial transcriptomics."""

import os, time, datetime, warnings
import numpy as np
from typing import Optional, Tuple, List
from anndata import AnnData
from sklearn.neighbors import BallTree, NearestNeighbors


# ============================================================================
# Stage 1: Multi-scale cell-type spatial descriptors
# ============================================================================

def compute_multiscale_descriptors(
    adata: AnnData,
    radii: Tuple[float, ...],
    cell_types: np.ndarray,
    cache_path: Optional[str] = None,
    slice_name: str = "slice",
    overwrite: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Multi-scale cell-type neighbourhood descriptor for every cell.

    For each cell i at each radius r, compute the normalised frequency
    distribution of ALL cell types within distance r:
        nd(i, r) in R^K,  sum_k nd(i, r)[k] = 1

    Stacking R radii -> (K*R)-dimensional descriptor.

    Properties
    ----------
    SE(2)-invariant:    built from pairwise distances, unchanged by rotation/translation.
    Cross-tp stable:    uses CELL TYPE labels, not raw gene expression.
    Locally distinctive: different tissue locations have different cell-type
                         neighbourhood patterns even in symmetric organs.
                         Repeated regions can mirror one another, but the
                         relative arrangement of cell types is subtly different
                         (different numbers of specific interneurons, different
                         layer thickness, corpus callosum asymmetry).

    Parameters
    ----------
    adata      : AnnData with .obsm['spatial'] and .obs['cell_type_annot']
    radii      : tuple of float -- e.g. (100., 200., 400.) in coordinate units
    cell_types : (K,) str array -- all cell types to include (union of A and B)
    cache_path : str or None
    slice_name : str
    overwrite  : bool
    verbose    : bool

    Returns
    -------
    desc : (n_cells, K * len(radii)) float32, L2-normalised rows
    """
    from tqdm import tqdm

    cf = None
    if cache_path is not None:
        os.makedirs(cache_path, exist_ok=True)
        r_str = "_".join(str(int(r)) for r in radii)
        cf = os.path.join(cache_path, f"msdesc_{slice_name}_{r_str}.npy")
        if os.path.exists(cf) and not overwrite:
            if verbose:
                print(f"[CAST desc] Loading cached: {cf}")
            return np.load(cf)

    coords = adata.obsm["spatial"].astype(np.float64)
    labels = np.asarray(adata.obs["cell_type_annot"].astype(str))
    K = len(cell_types)
    ct2idx = {c: i for i, c in enumerate(cell_types)}
    n = len(coords)
    R = len(radii)

    tree = BallTree(coords)
    desc = np.zeros((n, K * R), dtype=np.float32)

    if verbose:
        print(f"[CAST desc] {n} cells  K={K} types  {R} radii={radii}")

    for ri, radius in enumerate(radii):
        nbr_lists = tree.query_radius(coords, r=radius)
        for i in tqdm(range(n), desc=f"  r={radius}", disable=not verbose):
            nbrs = nbr_lists[i]
            if not len(nbrs):
                continue
            v = np.zeros(K, dtype=np.float32)
            for idx in nbrs:
                ct = labels[idx]
                if ct in ct2idx:
                    v[ct2idx[ct]] += 1.0
            s = v.sum()
            if s > 0:
                v /= s
            desc[i, ri * K : (ri + 1) * K] = v

    # L2-normalise so cosine similarity = dot product
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    desc /= norms

    if cf is not None:
        np.save(cf, desc)
        if verbose:
            print(f"[CAST desc] Saved: {cf}")

    return desc


# ============================================================================
# Stage 2: Candidate pair matching
# ============================================================================

def find_candidate_pairs(
    desc_A: np.ndarray,
    desc_B: np.ndarray,
    top_k: int = 10,
    min_score: float = 0.5,
    use_gpu: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each cell in A, find top_k most similar cells in B by cosine similarity.

    Since descriptors are L2-normalised, cosine similarity = dot product,
    computable as a single matrix multiplication -- fast on GPU.

    Parameters
    ----------
    desc_A, desc_B : (n, D) float32, L2-normalised
    top_k          : int -- candidates per A cell
    min_score      : float -- minimum similarity to retain as candidate
    use_gpu        : bool
    verbose        : bool

    Returns
    -------
    pair_i  : (M,) int32 -- indices into A
    pair_j  : (M,) int32 -- indices into B
    pair_sc : (M,) float32 -- similarity scores
    """
    from ._gpu import resolve_device, to_torch

    device = resolve_device(use_gpu)
    n_A, n_B = len(desc_A), len(desc_B)
    k_ = min(top_k, n_B)

    if verbose:
        print(f"[CAST pairs] Descriptor matching (n_A={n_A}, n_B={n_B}, top_k={k_}) ...")

    if device == "cuda":
        import torch
        dA = to_torch(desc_A, device, dtype=torch.float32)
        dB = to_torch(desc_B, device, dtype=torch.float32)
        S = dA @ dB.T
        vals, idxs = torch.topk(S, k=k_, dim=1)
        vals = vals.cpu().numpy().astype(np.float32)
        idxs = idxs.cpu().numpy().astype(np.int32)
    else:
        # Batched CPU matmul to avoid peak memory spike for 15k*15k
        batch = 1000
        vals = np.zeros((n_A, k_), dtype=np.float32)
        idxs = np.zeros((n_A, k_), dtype=np.int32)
        for start in range(0, n_A, batch):
            end = min(start + batch, n_A)
            S_b = desc_A[start:end] @ desc_B.T   # (batch, n_B)
            ki = min(k_, S_b.shape[1])
            part_idx = np.argpartition(-S_b, ki, axis=1)[:, :ki]
            part_val = S_b[np.arange(end - start)[:, None], part_idx]
            order = np.argsort(-part_val, axis=1)
            vals[start:end] = part_val[np.arange(end - start)[:, None], order]
            idxs[start:end] = part_idx[np.arange(end - start)[:, None], order]
    # Flatten, filter by min_score
    i_all = np.repeat(np.arange(n_A, dtype=np.int32), k_)
    j_all = idxs.ravel()
    s_all = vals.ravel()
    keep  = s_all >= min_score
    pair_i  = i_all[keep].astype(np.int32)
    pair_j  = j_all[keep].astype(np.int32)
    pair_sc = s_all[keep].astype(np.float32)

    # Sort by descending score
    order = np.argsort(-pair_sc)
    pair_i, pair_j, pair_sc = pair_i[order], pair_j[order], pair_sc[order]

    if verbose:
        print(f"[CAST pairs] {len(pair_i)} candidates (score >= {min_score})")

    return pair_i, pair_j, pair_sc


# ============================================================================
# Stage 3: RANSAC SE(2) estimation
# ============================================================================

def _se2_from_two_pairs(x1, x2, y1, y2):
    """
    Compute SE(2) from two point correspondences: x1->y1, x2->y2.
    Returns (R, t) or None if degenerate.
    """
    dA = x2 - x1
    dB = y2 - y1
    if np.linalg.norm(dA) < 1e-6 or np.linalg.norm(dB) < 1e-6:
        return None
    theta = np.arctan2(dB[1], dB[0]) - np.arctan2(dA[1], dA[0])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    t = ((y1 - R @ x1) + (y2 - R @ x2)) * 0.5
    return R, t


def ransac_se2(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_sc: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    n_iter: int = 2000,
    inlier_threshold: Optional[float] = None,
    min_inlier_frac: float = 0.05,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    RANSAC SE(2) from candidate cell-pair correspondences.

    Key property: the CORRECT transformation (A->correct region of B) maps
    ALL A cells near their true B counterparts. Wrong transformations
    (A->wrong repeated region) have near-zero spatial consistency.

    This is the mechanism that breaks repeated-region ambiguity without any
    assumption about the number of regions, organ type, or overlap fraction.

    Algorithm
    ---------
    For n_iter iterations:
      1. Sample 2 candidate pairs (weighted by descriptor similarity score)
      2. Compute SE(2) uniquely from 2 point correspondences
      3. Apply to all A cells, count cells that land near a B cell (inliers)
    Select best hypothesis; refine with weighted Procrustes on all inliers.

    Complexity: O(n_iter * n_A) -- parallelisable

    Parameters
    ----------
    pair_i, pair_j, pair_sc : candidate correspondences from find_candidate_pairs
    coords_A, coords_B      : (n, 2) float64 cell coordinates
    n_iter     : int   -- RANSAC iterations
    inlier_threshold : float or None -- None = 5% of B's spatial diameter
    min_inlier_frac  : float -- warn if below this
    verbose    : bool

    Returns
    -------
    R_best     : (2,2) float64
    t_best     : (2,)  float64
    n_inliers  : int
    inlier_mask: (n_A,) bool
    """
    n_A = len(coords_A)

    if inlier_threshold is None:
        diam = float(np.linalg.norm(
            coords_B.max(axis=0) - coords_B.min(axis=0)))
        inlier_threshold = max(0.02 * diam, 1.0)

    if verbose:
        print(f"[CAST RANSAC] {len(pair_i)} candidates  "
              f"{n_iter} iters  threshold={inlier_threshold:.1f}")

    probs = pair_sc.astype(np.float64)
    probs /= probs.sum()

    tree_B = BallTree(coords_B)

    R_best = np.eye(2)
    t_best = np.zeros(2)
    n_best = 0
    mask_best = np.zeros(n_A, dtype=bool)

    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(len(pair_i), size=2, replace=False, p=probs)
        i1, j1 = int(pair_i[idx[0]]), int(pair_j[idx[0]])
        i2, j2 = int(pair_i[idx[1]]), int(pair_j[idx[1]])

        if i1 == i2 or j1 == j2:
            continue

        res = _se2_from_two_pairs(
            coords_A[i1], coords_A[i2],
            coords_B[j1], coords_B[j2])
        if res is None:
            continue

        R_h, t_h = res
        cA_t = (R_h @ coords_A.T).T + t_h
        dists, _ = tree_B.query(cA_t, k=1)
        mask = dists.ravel() < inlier_threshold
        n_in = int(mask.sum())

        if n_in > n_best:
            n_best, mask_best = n_in, mask
            R_best, t_best = R_h, t_h

    # Refine on all inliers with weighted Procrustes
    if mask_best.sum() >= 3:
        cA_t = (R_best @ coords_A.T).T + t_best
        _, nn = tree_B.query(cA_t[mask_best], k=1)
        nn = nn.ravel()
        idxA = np.where(mask_best)[0]
        n_in = len(idxA)
        # Build soft plan: uniform weight on inlier pairs
        pi_ref = np.zeros((n_A, len(coords_B)), dtype=np.float64)
        pi_ref[idxA, nn] = 1.0 / n_in
        from .seot import weighted_procrustes
        R_best, t_best, _ = weighted_procrustes(pi_ref, coords_A, coords_B)
        # Recount inliers with refined transform
        cA_t = (R_best @ coords_A.T).T + t_best
        dists, _ = tree_B.query(cA_t, k=1)
        mask_best = dists.ravel() < inlier_threshold
        n_best = int(mask_best.sum())

    theta_best = float(np.degrees(np.arctan2(R_best[1, 0], R_best[0, 0])))
    inlier_frac = n_best / n_A

    if verbose:
        print(f"[CAST RANSAC] Best: theta={theta_best:.1f}  "
              f"inliers={n_best}/{n_A} ({inlier_frac*100:.1f}%)")

    if inlier_frac < min_inlier_frac:
        warnings.warn(
            f"[CAST RANSAC] Only {inlier_frac*100:.1f}% inliers -- "
            "transformation may be unreliable. "
            "Try larger n_iter, smaller min_score, or larger inlier_threshold.",
            stacklevel=3)

    return R_best, t_best, n_best, mask_best


# ============================================================================
# Public entry point
# ============================================================================

def pairwise_align_cast(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    filePath: str,
    # Descriptor
    radii: Optional[Tuple[float, ...]] = None,
    top_k_pairs: int = 10,
    min_desc_score: float = 0.5,
    use_lrf: bool = True,
    # RANSAC
    ransac_n_iter: int = 2000,
    inlier_threshold: Optional[float] = None,
    min_inlier_frac: float = 0.05,
    # SEOT EM
    max_em_iter: int = 50,
    tol_em: float = 1e-5,
    reg_sinkhorn: float = 0.01,
    rho_A: Optional[float] = None,
    rho_B: Optional[float] = None,
    base_rho: float = 0.5,
    # Cross-timepoint extras
    cvae_model=None,
    cvae_path: Optional[str] = None,
    cvae_epochs: int = 80,
    cvae_latent_dim: int = 32,
    cross_timepoint: bool = False,
    # LDDMM BCD (cross-timepoint spatial deformation)
    use_lddmm: bool = False,
    sigma_v: float = 300.0,
    lambda_v: float = 1.0,
    lddmm_lr: float = 0.01,
    lddmm_n_iter: int = 50,
    n_bcd_rounds: int = 3,
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
    CAST: Coarse-to-fine Anatomical Spatial Transcriptomics alignment.

    The generalised solution to the partial-overlap, bilateral-symmetry,
    arbitrary-SE(2) alignment problem. No assumptions about organ type,
    number of symmetric regions, or which slice is larger.

    Why previous methods fail
    -------------------------
    FGW/GW:  rotation-invariant -- cannot recover SE(2). Post-hoc centroid
             translation fails for partial/asymmetric overlap.

    BISPA:   community-level matching -- global cell-type distributions are
             IDENTICAL between repeated regions by definition. Cannot
             distinguish left from right at the global level.

    SEOT:    correct objective (explicit R,t) but relies on BISPA for initialisation
             which fails due to symmetry. Multi-start covers 8 rotations
             but without a strong spatial-consistency discriminator.

    CAST solves it by using SPATIAL CONSISTENCY as the discriminator
    ---------------------------------------------------------------
    The correct SE(2) transformation maps A cells near their true B counterparts.
    A wrong transformation (e.g. pointing to a different symmetric region) maps A cells
    to locations where B's spatial cell-type pattern is DIFFERENT -- even though
    the global histograms are symmetric, the local patterns are not.

    A cell in the left motor cortex has its unique neighbourhood (certain
    proportions of layer 2/3 neurons, inhibitory interneurons, etc. at radii
    100/200/400 um).  The best matching cell in B is its true counterpart --
    not its repeated-region counterpart, because the mirrored
    has a subtly different pattern (different interneuron density, different
    layer thickness, different proximity to corpus callosum).

    RANSAC counts how many A cells, when transformed by a candidate SE(2),
    land near a B cell. The correct SE(2) gets many inliers; any wrong SE(2)
    gets almost none.

    Same-timepoint pipeline (Stages 1-3)
    -------------------------------------
    Stage 1: Multi-scale cell-type descriptors (SE(2)-invariant, cross-tp stable)
             desc(i) = [nd(i, r), nd(i, 2r), nd(i, 4r)]  in R^(K*3)
             Locally distinctive: each location has a unique neighbourhood pattern.

    Stage 2: Candidate pair matching (cosine similarity in descriptor space)
             For each A cell, top-K most similar B cells.  O(n_A * n_B * K * 3).

    Stage 3: RANSAC SE(2) (spatial consistency filtering)
             Sample 2 pairs -> SE(2) hypothesis -> count spatial inliers.
             Select hypothesis with most inliers. Refine with Procrustes.
             THIS BREAKS SYMMETRY.

    Stage 4: SEOT EM (joint optimisation of R,t and cell correspondences)
             Polish the RANSAC result with partial unbalanced OT.

    Cross-timepoint extras (Stages 5-6)
    ------------------------------------
    Stage 5: cVAE latent cost for M_bio (handles temporal expression drift).
    Stage 6: LDDMM BCD (optional) -- models non-rigid spatial deformation.

    Generalisation
    --------------
    Works for any organ, any number of symmetric regions -- no organ-specific code.
    Works whether A or B is larger (symmetric treatment).
    Works for 30% to 95% overlap.
    Works for same-timepoint and cross-timepoint.

    Parameters
    ----------
    sliceA, sliceB : AnnData
        Both must have .obsm['spatial'] (n,2) and .obs['cell_type_annot'].
        Roles are symmetric -- no assumption about which is larger.
    alpha  : spatial weight in SEOT EM [0=biology only, 1=spatial only].
             Recommended: 0.5 for same-tp, 0.6 for cross-tp.
    beta   : cell-type mismatch weight in M_bio.
    gamma  : neighbourhood JSD weight.
    radius : neighbourhood radius (same units as .obsm['spatial']).
    filePath : directory for logs and cache files.

    radii : tuple of float or None
        Descriptor radii. None = (radius, 2*radius, 4*radius).
        Should span single-cell to regional scales.
    top_k_pairs : int, default 10
        Candidate matches per A cell in descriptor search.
    min_desc_score : float, default 0.5
        Minimum cosine similarity to retain as candidate.
    ransac_n_iter : int, default 2000
        RANSAC iterations. 2000 gives >99% confidence with 10% correct pairs.
    inlier_threshold : float or None
        Spatial inlier threshold. None = 2% of B's diameter (auto).
    min_inlier_frac : float, default 0.05
        Warn if fewer than this fraction are inliers.

    max_em_iter : int, default 50 -- SEOT EM iterations.
    reg_sinkhorn : float, default 0.01 -- Sinkhorn entropic reg.
    rho_A, rho_B : float or None -- KL marginal relaxation (None=auto).
    base_rho : float, default 0.5 -- scale for auto rho.

    cross_timepoint : bool, default False
        True -> use cVAE latent cost instead of raw expression cosine.
    use_lddmm : bool, default False
        True -> add LDDMM BCD after SEOT EM (for cross-tp spatial deformation).

    return_diagnostics : bool, default False
        True -> returns (pi, diagnostics_dict).

    Returns
    -------
    pi : (n_A_filt, n_B_filt) float64 -- transport plan on shared cell subset.
    If return_diagnostics=True:
        (pi, {R, t, theta_deg, n_inliers, inlier_frac, pi_mass,
              sliceA_aligned, residual_history, ...})
    """
    if use_lrf:
        from .cast_v2 import pairwise_align_cast_v2

        return pairwise_align_cast_v2(
            sliceA=sliceA,
            sliceB=sliceB,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            radius=radius,
            filePath=filePath,
            radii=radii,
            top_k_pairs=top_k_pairs,
            min_desc_score=min_desc_score,
            use_lrf=True,
            ransac_n_iter=ransac_n_iter,
            inlier_threshold=inlier_threshold,
            min_inlier_frac=min_inlier_frac,
            max_em_iter=max_em_iter,
            tol_em=tol_em,
            reg_sinkhorn=reg_sinkhorn,
            rho_A=rho_A,
            rho_B=rho_B,
            base_rho=base_rho,
            cvae_model=cvae_model,
            cvae_path=cvae_path,
            cvae_epochs=cvae_epochs,
            cvae_latent_dim=cvae_latent_dim,
            cross_timepoint=cross_timepoint,
            use_lddmm=use_lddmm,
            sigma_v=sigma_v,
            lambda_v=lambda_v,
            lddmm_lr=lddmm_lr,
            lddmm_n_iter=lddmm_n_iter,
            n_bcd_rounds=n_bcd_rounds,
            use_rep=use_rep,
            numItermax=numItermax,
            use_gpu=use_gpu,
            gpu_verbose=gpu_verbose,
            verbose=verbose,
            sliceA_name=sliceA_name,
            sliceB_name=sliceB_name,
            overwrite=overwrite,
            neighborhood_dissimilarity=neighborhood_dissimilarity,
            return_diagnostics=return_diagnostics,
        )

    import ot as pot
    from .seot import seot_em, weighted_procrustes
    from .core import _preprocess, _to_np

    start_time = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_cast_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name else f"{filePath}/log_cast.txt")
    log = open(log_name, "w")
    log.write(f"pairwise_align_cast -- INCENT-SE CAST\n{datetime.datetime.now()}\n")
    log.write(f"alpha={alpha}  beta={beta}  gamma={gamma}  radius={radius}\n"
              f"ransac_n_iter={ransac_n_iter}  cross_timepoint={cross_timepoint}\n\n")

    if radii is None:
        radii = (radius, 2.0 * radius, 4.0 * radius)

    # Shared cell types (union so descriptors have same dimension on both sides)
    ct_A = set(sliceA.obs["cell_type_annot"].astype(str).unique())
    ct_B = set(sliceB.obs["cell_type_annot"].astype(str).unique())
    shared_ct = np.array(sorted(ct_A | ct_B))

    coords_A_raw = np.asarray(sliceA.obsm["spatial"], dtype=np.float64)
    coords_B_raw = np.asarray(sliceB.obsm["spatial"], dtype=np.float64)

    # ==================================================================
    # STAGE 1: Multi-scale cell-type descriptors
    # ==================================================================
    print("[CAST] Stage 1: Multi-scale descriptors ...")
    desc_A = compute_multiscale_descriptors(
        sliceA, radii=radii, cell_types=shared_ct,
        cache_path=filePath,
        slice_name=f"{sliceA_name or 'A'}_cast",
        overwrite=overwrite, verbose=gpu_verbose)

    desc_B = compute_multiscale_descriptors(
        sliceB, radii=radii, cell_types=shared_ct,
        cache_path=filePath,
        slice_name=f"{sliceB_name or 'B'}_cast",
        overwrite=overwrite, verbose=gpu_verbose)

    log.write(f"Descriptors: A{desc_A.shape}  B{desc_B.shape}  radii={radii}\n")

    # ==================================================================
    # STAGE 2: Candidate matching
    # ==================================================================
    print("[CAST] Stage 2: Candidate pair matching ...")
    pair_i, pair_j, pair_sc = find_candidate_pairs(
        desc_A, desc_B, top_k=top_k_pairs,
        min_score=min_desc_score, use_gpu=use_gpu, verbose=gpu_verbose)

    if len(pair_i) < 4:
        raise ValueError(
            f"[CAST] Only {len(pair_i)} candidate pairs found. "
            "Try reducing min_desc_score or increasing top_k_pairs.")

    log.write(f"Candidate pairs: {len(pair_i)}\n")

    # ==================================================================
    # STAGE 3: RANSAC -- spatial consistency breaks bilateral symmetry
    # ==================================================================
    print("[CAST] Stage 3: RANSAC SE(2) ...")
    R_ransac, t_ransac, n_inliers, inlier_mask = ransac_se2(
        pair_i, pair_j, pair_sc,
        coords_A_raw, coords_B_raw,
        n_iter=ransac_n_iter,
        inlier_threshold=inlier_threshold,
        min_inlier_frac=min_inlier_frac,
        verbose=gpu_verbose)

    inlier_frac = n_inliers / len(coords_A_raw)
    theta_ransac = float(np.degrees(np.arctan2(R_ransac[1, 0], R_ransac[0, 0])))
    log.write(f"RANSAC: theta={theta_ransac:.1f}  "
              f"inliers={n_inliers}/{len(coords_A_raw)} ({inlier_frac:.3f})\n")
    print(f"[CAST] RANSAC: theta={theta_ransac:.1f}  "
          f"inliers={n_inliers}/{len(coords_A_raw)} ({inlier_frac*100:.1f}%)")

    # ==================================================================
    # STAGE 4: Build M_bio and SEOT EM
    # ==================================================================
    print("[CAST] Stage 4: Building M_bio ...")

    model = None

    if cross_timepoint:
        from .cvae import INCENT_cVAE, train_cvae, latent_cost
        if cvae_model is not None:
            model = cvae_model
        elif cvae_path is not None and os.path.exists(cvae_path):
            model = INCENT_cVAE.load(cvae_path)
        else:
            print("[CAST] Training cVAE ...")
            model = train_cvae([sliceA, sliceB],
                               latent_dim=cvae_latent_dim,
                               epochs=cvae_epochs,
                               verbose=gpu_verbose)
            if cvae_path:
                model.save(cvae_path)

    # Apply RANSAC transform to sliceA so INCENT preprocessing
    # computes neighbourhood distributions in the correct coordinate frame
    sliceA_rough = sliceA.copy()
    sliceA_rough.obsm["spatial"] = (
        (R_ransac @ coords_A_raw.T).T + t_ransac)

    log2 = open(f"{filePath}/log_cast_pre.txt", "w")
    p = _preprocess(
        sliceA_rough, sliceB, alpha, beta, gamma, radius, filePath,
        use_rep, None, None, None,
        numItermax, pot.backend.NumpyBackend(), use_gpu, gpu_verbose,
        sliceA_name, sliceB_name, overwrite, neighborhood_dissimilarity,
        log2)
    log2.close()

    sA_filt = p["sliceA"]
    sB_filt = p["sliceB"]
    a_np    = _to_np(p["a"])
    b_np    = _to_np(p["b"])
    n_A, n_B = sA_filt.shape[0], sB_filt.shape[0]

    if cross_timepoint:
        from .cvae import latent_cost
        if model is None:
            raise RuntimeError("cVAE model was not initialised for cross_timepoint alignment")
        M1_np = latent_cost(sA_filt, sB_filt, model).astype(np.float32)
    else:
        M1_np = _to_np(p["cosine_dist_gene_expr"]).astype(np.float32)

    M2_np = _to_np(p["M2"]).astype(np.float32)

    from .topology import compute_fingerprints, fingerprint_cost
    fp_A = compute_fingerprints(sA_filt, radius=radius, n_bins=16,
                                 cache_path=filePath,
                                 slice_name=f"{sliceA_name or 'A'}_cast",
                                 overwrite=overwrite, verbose=gpu_verbose)
    fp_B = compute_fingerprints(sB_filt, radius=radius, n_bins=16,
                                 cache_path=filePath,
                                 slice_name=f"{sliceB_name or 'B'}_cast",
                                 overwrite=overwrite, verbose=gpu_verbose)
    M_topo = fingerprint_cost(fp_A, fp_B, metric="cosine",
                               use_gpu=use_gpu).astype(np.float32)

    M_bio = M1_np + gamma * M2_np + 0.3 * M_topo   # (n_A, n_B) float32

    # Coordinates in the RANSAC-corrected frame (already applied above)
    coords_A = sA_filt.obsm["spatial"].astype(np.float64)
    coords_B = sB_filt.obsm["spatial"].astype(np.float64)

    # Size-ratio rho (no assumption about which slice is larger)
    n_A_total = len(sliceA)
    n_B_total = len(sliceB)
    size_ratio = float(n_A_total) / float(n_B_total)
    rho_A_use = rho_A if rho_A is not None else float(
        base_rho * min(float(n_B_total) / max(n_A_total, 1), 1.0))
    rho_B_use = rho_B if rho_B is not None else float(
        base_rho * min(size_ratio, 1.0))

    log.write(f"SEOT rho_A={rho_A_use:.4f}  rho_B={rho_B_use:.4f}  "
              f"size_ratio={size_ratio:.3f}\n")
    print(f"[CAST] SEOT EM: rho_A={rho_A_use:.3f}  rho_B={rho_B_use:.3f}  "
          f"size_ratio={size_ratio:.3f}")

    print("[CAST] Stage 4: SEOT EM ...")
    pi, R_em, t_em, history, _ = seot_em(
        M_bio=M_bio,
        coords_A=coords_A,
        coords_B=coords_B,
        a=a_np, b=b_np,
        R_init=np.eye(2),     # RANSAC already applied to coords_A
        t_init=np.zeros(2),
        alpha=alpha,
        rho_A=rho_A_use,
        rho_B=rho_B_use,
        reg_sinkhorn=reg_sinkhorn,
        max_iter=max_em_iter,
        tol=tol_em,
        verbose=verbose,
    )

    # Compose total transformation: RANSAC * EM refinement
    R_total = R_em @ R_ransac
    t_total = R_em @ t_ransac + t_em
    theta_total = float(np.degrees(np.arctan2(R_total[1, 0], R_total[0, 0])))

    # ==================================================================
    # STAGE 5 (cross-timepoint): LDDMM BCD for spatial deformation
    # ==================================================================
    phi = None
    if use_lddmm and cross_timepoint:
        print("[CAST] Stage 5: LDDMM BCD (spatial deformation) ...")
        D_B_current = _to_np(p["D_B"])
        D_A_np      = _to_np(p["D_A"])
        a_np_bcd    = a_np.copy()
        b_np_bcd    = b_np.copy()

        for bcd_round in range(1, n_bcd_rounds + 1):
            from .lddmm import estimate_deformation, deformed_distances
            phi = estimate_deformation(
                pi, coords_A, coords_B,
                sigma_v=sigma_v, lambda_v=lambda_v,
                lr=lddmm_lr, n_iter=lddmm_n_iter,
                use_gpu=use_gpu, verbose=False)

            D_B_current = deformed_distances(
                coords_B, phi, normalise=True, use_gpu=use_gpu)

            import ot.gromov
            alpha_fugw = float((1.0 - alpha) / alpha) if 1e-6 < alpha < 1 - 1e-6 else 1.0
            pi_s, _ = ot.gromov.fused_unbalanced_gromov_wasserstein(
                Cx=D_A_np, Cy=D_B_current,
                wx=a_np_bcd, wy=b_np_bcd,
                reg_marginals=(rho_A_use, rho_B_use),
                epsilon=reg_sinkhorn, divergence="kl",
                unbalanced_solver="sinkhorn",
                alpha=alpha_fugw, M=M_bio.astype(np.float64),
                max_iter=50, tol=1e-6, max_iter_ot=500, tol_ot=1e-6,
                log=False, verbose=False)
            pi = np.asarray(pi_s, dtype=np.float64)
            print(f"[CAST LDDMM] BCD round {bcd_round}/{n_bcd_rounds}  "
                  f"pi_mass={pi.sum():.4f}")

    pi_mass = float(pi.sum())
    runtime = time.time() - start_time
    log.write(f"Final: theta={theta_total:.2f}  pi_mass={pi_mass:.4f}  "
              f"Runtime={runtime:.1f}s\n")
    log.close()

    print(f"[CAST] Done.  theta={theta_total:.1f}  "
          f"t=({t_total[0]:.1f},{t_total[1]:.1f})  "
          f"pi_mass={pi_mass:.4f}  Runtime={runtime:.1f}s")

    sliceA_aligned = sliceA.copy()
    sliceA_aligned.obsm["spatial"] = (
        (R_total @ sliceA.obsm["spatial"].astype(np.float64).T).T + t_total)

    if return_diagnostics:
        return pi, {
            "R":                R_total,
            "t":                t_total,
            "theta_deg":        theta_total,
            "n_inliers":        n_inliers,
            "inlier_frac":      inlier_frac,
            "pi_mass":          pi_mass,
            "sliceA_aligned":   sliceA_aligned,
            "ransac_R":         R_ransac,
            "ransac_t":         t_ransac,
            "residual_history": history,
            "phi":              phi,
        }
    return pi
