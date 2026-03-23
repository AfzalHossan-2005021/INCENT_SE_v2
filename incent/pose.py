"""
pose.py — SE(2) Pose Estimator for INCENT-SE   [v2 — robust to large/offset coords]
====================================================================================
Recovers the in-plane rigid transformation (rotation angle θ, translation t)
between two spatial transcriptomics slices **before** running any optimal
transport.

Why do this first?
------------------
FGW-based alignment is inherently rotation-invariant: it compares
*pairwise distances*, so it finds the best coupling regardless of orientation.
That is useful — but it means the algorithm can never tell you *where* slice A
sits in the coordinate frame of slice B.  Worse, for bilaterally symmetric
tissues (e.g. brain) there are two orientations that achieve nearly identical
FGW cost, and the optimizer picks one essentially at random.

Strategy — Fourier-Mellin Transform (FMT)
-----------------------------------------
For each cell type k we build a 2-D spatial density image ρ_k(x,y).
The key fact (Fourier shift / scale theorem):

    |DFT( f rotated by θ )| = |DFT(f)| rotated by θ

So rotation in image space becomes *rotation in frequency space*.
Mapping the magnitude spectrum to log-polar coordinates converts that
rotation into a **horizontal translation**, which we find cheaply via
normalized cross-correlation (NCC).

Critical fix — coordinate normalisation
----------------------------------------
MERFISH and similar technologies store coordinates in microns, nanometres,
or arbitrary scanner units.  A typical MERFISH brain dataset has x,y values
in the range 0–150 000 μm.  Two slices may sit in entirely different
regions of that space (e.g. sliceA at x ∈ [0, 5000], sliceB at
x ∈ [120000, 145000]).  If we rasterise both onto a *shared* bounding box
the cells each occupy a tiny corner of the grid, giving near-zero Fourier
signals and meaningless NCC scores.  The phase-correlation step then reports
a spurious translation of ~140 000 units.

Fix: centre every slice around its own centroid (μ_x, μ_y) before
rasterisation.  Rotation is centroid-invariant so this does not affect θ.
Translation is recovered analytically from the centroid offset:
    t = centroid_B − R(θ) · centroid_A
This is exact when both centroids correspond to the same physical tissue
location; for partial-overlap slices it provides the coarse alignment that
the OT solver refines at cell resolution.

Public API
----------
estimate_pose(sliceA, sliceB, **kwargs) -> (theta_deg, tx, ty, score)
apply_pose(sliceA, theta_deg, tx, ty)  -> AnnData  (coords transformed, copy)
"""

import numpy as np
import warnings
from typing import Tuple, Optional
from anndata import AnnData
from scipy.signal import correlate2d


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_matrix(theta_deg: float) -> np.ndarray:
    """
    Build a 2×2 counter-clockwise rotation matrix.

    Parameters
    ----------
    theta_deg : float — angle in degrees.

    Returns
    -------
    R : (2, 2) float64.
    """
    rad = np.deg2rad(theta_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s], [s, c]])


def _centre_coords(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Shift a point cloud so its centroid is at the origin.

    Returns
    -------
    centred   : (n, 2) — zero-centred coordinates.
    centroid  : (2,)   — the original centroid (subtracted).
    half_span : float  — half of max(x_range, y_range); used for grid scaling.
    """
    centroid  = coords.mean(axis=0)
    centred   = coords - centroid
    x_range   = coords[:, 0].max() - coords[:, 0].min()
    y_range   = coords[:, 1].max() - coords[:, 1].min()
    half_span = max(x_range, y_range) / 2.0 + 1e-6
    return centred, centroid, half_span


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Rasterise cell-type density on a centred, normalised grid
# ─────────────────────────────────────────────────────────────────────────────

def _rasterise_density_centred(
    coords_centred: np.ndarray,
    labels: np.ndarray,
    cell_types: np.ndarray,
    grid_size: int,
    half_span: float,
    sigma_px: float = 2.5,
) -> np.ndarray:
    """
    Rasterise centred cell coordinates into a (K, grid_size, grid_size) density stack.

    The grid covers [-half_span, +half_span] in both x and y.
    Cells are Gaussian-smoothed so the density field is continuous.

    Parameters
    ----------
    coords_centred : (n, 2) float — coordinates already centred at origin.
    labels         : (n,) str    — cell-type label per cell.
    cell_types     : (K,) str    — unique cell types to rasterise.
    grid_size      : int         — pixel resolution (e.g. 256).
    half_span      : float       — world-coordinate half-width of the grid.
    sigma_px       : float       — Gaussian smoothing radius (pixels).

    Returns
    -------
    density : (K, grid_size, grid_size) float32.
        Each channel is L2-normalised to 1 so every cell type contributes
        equally regardless of its abundance.
    """
    from scipy.ndimage import gaussian_filter

    K       = len(cell_types)
    density = np.zeros((K, grid_size, grid_size), dtype=np.float32)
    G       = grid_size
    centre  = G // 2

    # Map centred world coords → pixel indices
    # half_span world units spans half the grid (centre pixels)
    scale   = (G / 2.0 - 1) / half_span     # pixels per world unit
    px      = (coords_centred[:, 0] * scale + centre).astype(int)
    py      = (coords_centred[:, 1] * scale + centre).astype(int)
    # Clamp — cells near the edge may round slightly outside
    px      = np.clip(px, 0, G - 1)
    py      = np.clip(py, 0, G - 1)

    for k, ct in enumerate(cell_types):
        mask = labels == ct
        if mask.sum() == 0:
            continue
        img  = np.zeros((G, G), dtype=np.float32)
        np.add.at(img, (py[mask], px[mask]), 1.0)
        img  = gaussian_filter(img, sigma=sigma_px)
        norm = np.linalg.norm(img)
        if norm > 1e-10:
            img /= norm
        density[k] = img

    return density


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Log-polar Fourier spectrum
# ─────────────────────────────────────────────────────────────────────────────

def _log_polar_spectrum(density_stack: np.ndarray, num_angles: int = 360) -> np.ndarray:
    """
    Compute the log-polar Fourier magnitude spectrum, averaged across channels.

    Why log-polar?
    --------------
    In the Fourier magnitude spectrum, a rotation of the image becomes a
    rotation of the spectrum.  Mapping to log-polar coordinates turns that
    rotation into a *translation* in the angle axis — making NCC the right
    tool for detecting it.

    Non-empty channels (cell types that are actually present) are averaged
    so every channel contributes equally.

    Parameters
    ----------
    density_stack : (K, H, W) float32 — K cell-type density images.
    num_angles    : int — angular resolution (bins).

    Returns
    -------
    lp : (num_angles, log_r_bins) float32 — averaged log-polar spectrum.
    """
    from scipy.ndimage import map_coordinates

    K, H, W  = density_stack.shape
    cy, cx   = H // 2, W // 2
    lp_sum   = None
    n_active = 0

    for k in range(K):
        img = density_stack[k]
        if img.max() < 1e-10:      # skip empty cell-type channels
            continue

        # FFT magnitude spectrum, log-compressed, DC at centre
        mag = np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float64))))
        mag = np.log1p(mag)

        max_r      = min(cy, cx)
        log_r_bins = max(int(np.log2(max(max_r, 2)) * 8) + 1, 4)
        angles     = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        log_rs     = np.logspace(0, np.log10(max(max_r, 1.1)), log_r_bins)

        rs_grid, a_grid = np.meshgrid(log_rs, angles)
        yy = cy + rs_grid * np.sin(a_grid)
        xx = cx + rs_grid * np.cos(a_grid)

        lp_k = map_coordinates(mag, [yy.ravel(), xx.ravel()],
                               order=1, mode='constant').reshape(num_angles, log_r_bins)
        if lp_sum is None:
            lp_sum = lp_k.copy()
        else:
            lp_sum += lp_k
        n_active += 1

    if lp_sum is None or n_active == 0:
        # All channels empty — return a flat spectrum (NCC will give θ=0)
        lp_sum = np.zeros((num_angles, 4), dtype=np.float32)
    else:
        lp_sum /= n_active

    return lp_sum


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: NCC peak → rotation angle
# ─────────────────────────────────────────────────────────────────────────────

def _ncc_peak_angle(lp_A: np.ndarray, lp_B: np.ndarray, num_angles: int = 360) -> float:
    """
    Find the rotation angle between two log-polar spectra via NCC.

    A rotation ↔ angular shift in log-polar space.
    The NCC peak shift gives θ (mod 180° due to Fourier symmetry).

    Parameters
    ----------
    lp_A, lp_B : (num_angles, log_r_bins) — log-polar spectra.
    num_angles  : int — bins in the angular axis.

    Returns
    -------
    theta_deg : float ∈ [0, 180) — estimated rotation angle.
    """
    # Collapse log-r → 1-D angular signal
    sig_A = lp_A.mean(axis=1)
    sig_B = lp_B.mean(axis=1)

    # Zero-mean normalise each signal before NCC
    def _norm(v):
        v = v - v.mean()
        s = v.std()
        return v / (s + 1e-10)

    ncc   = correlate2d(_norm(sig_A)[None, :],
                        _norm(sig_B)[None, :], mode='full')[0]
    shift = int(np.argmax(ncc)) - (num_angles - 1)
    theta = (shift * 360.0 / num_angles) % 180.0
    return float(theta)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Alignment score — Pearson correlation of centred density images
# ─────────────────────────────────────────────────────────────────────────────

def _alignment_score(
    coords_A_centred: np.ndarray,
    labels_A: np.ndarray,
    coords_B_centred: np.ndarray,
    labels_B: np.ndarray,
    cell_types: np.ndarray,
    grid_size: int,
    half_span: float,
) -> float:
    """
    Score how well two centred density stacks overlap (Pearson r, averaged).

    Both coordinate sets must already be centred at the origin and use
    the same half_span so the pixel scale is identical.

    Returns
    -------
    score : float ∈ [0, 1].
    """
    from scipy.stats import pearsonr

    d_A = _rasterise_density_centred(
        coords_A_centred, labels_A, cell_types, grid_size, half_span)
    d_B = _rasterise_density_centred(
        coords_B_centred, labels_B, cell_types, grid_size, half_span)

    score, n = 0.0, 0
    for k in range(len(cell_types)):
        a, b = d_A[k].ravel(), d_B[k].ravel()
        if a.std() > 1e-10 and b.std() > 1e-10:
            r, _ = pearsonr(a, b)
            if np.isfinite(r):
                score += max(float(r), 0.0)
                n     += 1

    return score / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION: estimate_pose
# ─────────────────────────────────────────────────────────────────────────────

def estimate_pose(
    sliceA: AnnData,
    sliceB: AnnData,
    grid_size: int = 256,
    sigma_px: float = 2.5,
    num_angles: int = 360,
    verbose: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Estimate the SE(2) rigid transformation (θ, tx, ty) mapping sliceA → sliceB.

    This is INCENT-SE Stage 1.  Runs in O(N_px · log N_px) — much cheaper
    than optimal transport — and provides a good initialisation so the FGW
    solver never gets stuck in the bilateral-symmetry local minimum.

    Algorithm
    ---------
    1. Centre sliceA and sliceB independently around their own centroids.
       This makes Fourier spectra comparable regardless of world-coordinate
       offsets (the fix for huge tx/ty when slices sit in different regions).
    2. Rasterise each cell-type channel onto a grid_size × grid_size image.
    3. Compute log-polar Fourier magnitude spectrum for each.
    4. Normalised cross-correlation in log-polar space → θ (mod 180°).
    5. Test both candidates θ and θ+180° by Pearson scoring; pick the better one.
    6. Compute translation analytically: t = centroid_B − R(θ) · centroid_A.

    Parameters
    ----------
    sliceA : AnnData
        Source slice.  Must have ``.obsm['spatial']`` (n_A, 2) and
        ``.obs['cell_type_annot']``.
    sliceB : AnnData
        Target slice.  Same requirements.
    grid_size : int, default 256
        Density image resolution.  256 is fast (~0.3 s); use 512 for finer
        angular resolution on complex slice geometries.
    sigma_px : float, default 2.5
        Gaussian smoothing radius in pixels.  Larger = smoother spectrum
        but slightly lower angular precision.
    num_angles : int, default 360
        Angular resolution of the log-polar grid (1° per bin).
    verbose : bool, default True

    Returns
    -------
    theta_deg : float — estimated CCW rotation angle (degrees).
    tx, ty    : float — translation in the same units as .obsm['spatial'].
    score     : float ∈ [0, 1] — alignment quality.
        > 0.3: reliable rotation estimate.
        < 0.15: treat rotation as uncertain; the centroid translation is
                still applied and will coarsely pre-align the slices.

    Examples
    --------
    >>> theta, tx, ty, score = estimate_pose(sliceA, sliceB)
    >>> sliceA_aligned = apply_pose(sliceA, theta, tx, ty)
    """
    if verbose:
        print("[pose] Rasterising cell-type density fields …")

    labels_A  = np.asarray(sliceA.obs['cell_type_annot'].astype(str))
    labels_B  = np.asarray(sliceB.obs['cell_type_annot'].astype(str))
    cell_types = np.intersect1d(np.unique(labels_A), np.unique(labels_B))

    if len(cell_types) == 0:
        raise ValueError("No shared cell types between slices.")

    coords_A = sliceA.obsm['spatial'].copy().astype(np.float64)
    coords_B = sliceB.obsm['spatial'].copy().astype(np.float64)

    # ── Centre each slice independently ─────────────────────────────────────
    # KEY FIX: prevents the huge-tx/ty bug when slices sit in different
    # regions of world space.  Rotation is centroid-invariant so θ is exact;
    # translation is recovered analytically below.
    cA, centroid_A, span_A = _centre_coords(coords_A)
    cB, centroid_B, span_B = _centre_coords(coords_B)
    half_span = max(span_A, span_B)     # common scale so pixel ↔ world is consistent

    # ── Rasterise ────────────────────────────────────────────────────────────
    density_A = _rasterise_density_centred(
        cA, labels_A, cell_types, grid_size, half_span, sigma_px)
    density_B = _rasterise_density_centred(
        cB, labels_B, cell_types, grid_size, half_span, sigma_px)

    # ── Log-polar Fourier spectra ─────────────────────────────────────────────
    if verbose:
        print("[pose] Computing log-polar Fourier spectra …")
    lp_A = _log_polar_spectrum(density_A, num_angles)
    lp_B = _log_polar_spectrum(density_B, num_angles)

    # ── Rotation via NCC ──────────────────────────────────────────────────────
    if verbose:
        print("[pose] Estimating rotation via NCC …")
    theta0 = _ncc_peak_angle(lp_A, lp_B, num_angles)
    theta1 = (theta0 + 180.0) % 360.0

    # ── Score both candidates ─────────────────────────────────────────────────
    R0   = _rotation_matrix(theta0)
    R1   = _rotation_matrix(theta1)
    cA0  = (R0 @ cA.T).T
    cA1  = (R1 @ cA.T).T
    sc0  = _alignment_score(cA0, labels_A, cB, labels_B, cell_types, grid_size, half_span)
    sc1  = _alignment_score(cA1, labels_A, cB, labels_B, cell_types, grid_size, half_span)

    if sc0 >= sc1:
        theta_best, best_score = theta0, sc0
        if verbose:
            print(f"[pose] θ={theta0:.1f}° (score={sc0:.3f})  "
                  f"vs θ+180={theta1:.1f}° (score={sc1:.3f})  → chose {theta0:.1f}°")
    else:
        theta_best, best_score = theta1, sc1
        if verbose:
            print(f"[pose] θ+180={theta1:.1f}° (score={sc1:.3f})  "
                  f"vs θ={theta0:.1f}° (score={sc0:.3f})  → chose {theta1:.1f}°")

    # ── Translation from centroid offset ──────────────────────────────────────
    # t = centroid_B − R(θ) · centroid_A
    # Rotating sliceA by θ and then adding t maps centroid_A to centroid_B.
    R_best = _rotation_matrix(theta_best)
    t_vec  = centroid_B - R_best @ centroid_A
    tx, ty = float(t_vec[0]), float(t_vec[1])

    if verbose:
        print(f"[pose] Done. θ={theta_best:.2f}°  tx={tx:.2f}  ty={ty:.2f}  "
              f"score={best_score:.3f}")
        if best_score < 0.15:
            warnings.warn(
                "[pose] Low Fourier alignment score (< 0.15).  Possible causes:\n"
                "  • Only one or very few cell types are shared between slices.\n"
                "  • Slices are from very different brain regions.\n"
                "  • Too few cells (< 200) in one slice.\n"
                "The rotation estimate may be unreliable, but the centroid-based\n"
                "translation (tx, ty) is still a valid coarse alignment.",
                stacklevel=2,
            )

    return float(theta_best), tx, ty, float(best_score)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION: apply_pose
# ─────────────────────────────────────────────────────────────────────────────

def apply_pose(
    sliceA: AnnData,
    theta_deg: float,
    tx: float,
    ty: float,
    inplace: bool = False,
) -> AnnData:
    """
    Apply an SE(2) rigid transformation to sliceA's spatial coordinates.

    Transformation:  x_new = R(θ) · x + t

    Parameters
    ----------
    sliceA    : AnnData — the slice to transform.
    theta_deg : float   — counter-clockwise rotation in degrees.
    tx, ty    : float   — translation in .obsm['spatial'] units.
    inplace   : bool, default False
        If True, modifies sliceA.obsm['spatial'] in-place.
        If False (default), returns a copy with transformed coordinates.

    Returns
    -------
    AnnData — with updated .obsm['spatial'].

    Examples
    --------
    >>> theta, tx, ty, score = estimate_pose(sliceA, sliceB)
    >>> sliceA_aligned = apply_pose(sliceA, theta, tx, ty)
    """
    if not inplace:
        sliceA = sliceA.copy()
    R      = _rotation_matrix(theta_deg)
    coords = sliceA.obsm['spatial'].astype(np.float64)
    sliceA.obsm['spatial'] = (R @ coords.T).T + np.array([tx, ty])
    return sliceA