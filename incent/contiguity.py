"""
contiguity.py — Spatial Contiguity Regulariser for INCENT-SE
=============================================================
Enforces that the matched cells form a spatially contiguous region rather
than scattered isolated points.

The regulariser
---------------
R_spatial(π) = <W_A, π D_B π^T>

where W_A[i,i'] = exp(-d_A(i,i')/σ) is a sparse local affinity matrix.
Nearby cells in A (high W_A) must be matched to nearby cells in B (small D_B).

Gradient: ∂R/∂π = 2 · W_A · π · D_B   (both W_A and D_B symmetric)

GPU acceleration
----------------
``build_spatial_affinity`` uses sklearn NearestNeighbors and stays on CPU
(no GPU k-NN library required).  The resulting sparse matrix is a small
construction step done once and cached.

``contiguity_gradient`` and ``contiguity_regulariser`` contain three chained
matrix products (W_A @ π) @ D_B and are GPU-accelerated when ``use_gpu=True``:
  - W_A (scipy CSR) is converted to torch sparse CSR on the GPU.
  - π and D_B (dense numpy) are moved to the GPU as float32 tensors.
  - The sparse × dense product W_A @ π is computed with torch.mm.
  - The dense × dense product (W_A @ π) @ D_B completes on GPU.

For n=15k this is ~3 large matrix products vs the CPU baseline.

Public API
----------
build_spatial_affinity(coords, sigma, k_nn) -> scipy.sparse.csr_matrix
contiguity_regulariser(pi, W_A, D_B, use_gpu) -> float
contiguity_gradient(pi, W_A, D_B, use_gpu)    -> (n_A, n_B) np.ndarray
augment_fgw_gradient(pi, W_A, D_B, lambda_spatial, use_gpu) -> (n_A, n_B) np.ndarray
estimate_overlap_fraction(pi, a, b) -> float
"""

import numpy as np
import scipy.sparse as sp
from ._gpu import resolve_device, to_torch, sparse_to_torch


# ─────────────────────────────────────────────────────────────────────────────
# Build sparse affinity (CPU only — sklearn NearestNeighbors)
# ─────────────────────────────────────────────────────────────────────────────

def build_spatial_affinity(
    coords: np.ndarray,
    sigma: float,
    k_nn: int = 20,
) -> sp.csr_matrix:
    """
    Build the sparse spatial affinity matrix W_A for sliceA.

    W_A[i, i'] = exp(-d(i,i') / σ)  if i' is among the k_nn nearest
                                      neighbours of i, else 0.

    This matrix encodes local spatial proximity.  It is computed once on CPU
    and passed to ``contiguity_gradient`` / ``contiguity_regulariser`` which
    move it to GPU as needed.

    Parameters
    ----------
    coords : (n_A, 2) float — spatial coordinates of cells in sliceA.
    sigma  : float — affinity decay length.  Good default: radius / 3.
    k_nn   : int, default 20 — neighbours per cell (sparse structure).

    Returns
    -------
    W_A : (n_A, n_A) scipy.sparse.csr_matrix, float32, symmetric.
    """
    from sklearn.neighbors import NearestNeighbors

    n   = len(coords)
    nn  = NearestNeighbors(n_neighbors=k_nn + 1, algorithm='ball_tree').fit(coords)
    dists, indices = nn.kneighbors(coords)
    dists   = dists[:, 1:]     # drop self (distance=0)
    indices = indices[:, 1:]

    rows = np.repeat(np.arange(n), k_nn)
    cols = indices.ravel()
    vals = np.exp(-dists.ravel() / (sigma + 1e-10)).astype(np.float32)

    W    = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    W    = (W + W.T) * 0.5       # symmetrise
    return W


# ─────────────────────────────────────────────────────────────────────────────
# Regulariser value — GPU-accelerated
# ─────────────────────────────────────────────────────────────────────────────

def contiguity_regulariser(
    pi: np.ndarray,
    W_A: sp.csr_matrix,
    D_B: np.ndarray,
    use_gpu: bool = False,
) -> float:
    """
    Evaluate R_spatial(π) = <W_A, π D_B π^T>.

    A low value means nearby cells in A are matched to nearby cells in B
    (contiguous overlap).  A high value means fragmented matching.

    Parameters
    ----------
    pi      : (n_A, n_B) float — current transport plan.
    W_A     : (n_A, n_A) scipy CSR — spatial affinity (from build_spatial_affinity).
    D_B     : (n_B, n_B) float — shared-scale normalised pairwise distances in B.
    use_gpu : bool, default False.

    Returns
    -------
    float — regulariser value ≥ 0.
    """
    device = resolve_device(use_gpu)

    if device == 'cuda':
        import torch
        pi_t  = to_torch(pi,  device, dtype=torch.float32)
        D_B_t = to_torch(D_B, device, dtype=torch.float32)
        W_t   = sparse_to_torch(W_A, device, dtype=torch.float32)

        pi_DB     = pi_t @ D_B_t        # (n_A, n_B)
        pi_DB_piT = pi_DB @ pi_t.T      # (n_A, n_A)
        # <W_A, M> via sparse element-wise product trick: W·M computed as sparse@dense
        # Equivalent: sum of (W_A * pi_DB_piT) = trace(W_A.T @ pi_DB_piT)
        val = torch.mm(W_t, pi_DB_piT).diagonal().sum()
        return float(val.item())

    # ── CPU path ───────────────────────────────────────────────────────────
    pi_DB     = pi @ D_B
    pi_DB_piT = pi_DB @ pi.T
    return float(W_A.multiply(pi_DB_piT).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Gradient — GPU-accelerated
# ─────────────────────────────────────────────────────────────────────────────

def contiguity_gradient(
    pi: np.ndarray,
    W_A: sp.csr_matrix,
    D_B: np.ndarray,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Compute ∂R_spatial/∂π = 2 · W_A · π · D_B.

    Derivation
    ----------
    R = <W_A, π D_B π^T>.  Differentiating w.r.t. π[i,j] and using the
    symmetry of W_A and D_B gives:
        ∂R/∂π = 2 · W_A · π · D_B

    GPU path
    --------
    W_A (scipy CSR) → torch sparse CSR on GPU.
    Sparse × dense:  W_A @ π  (torch.mm with sparse).
    Dense × dense:   (W_A @ π) @ D_B.
    Result moved back to CPU numpy.

    Parameters
    ----------
    pi      : (n_A, n_B) float — current transport plan.
    W_A     : (n_A, n_A) scipy CSR — spatial affinity.
    D_B     : (n_B, n_B) float — pairwise distances in B.
    use_gpu : bool, default False.

    Returns
    -------
    grad : (n_A, n_B) float64 numpy array.
    """
    device = resolve_device(use_gpu)

    if device == 'cuda':
        import torch
        pi_t  = to_torch(pi,  device, dtype=torch.float32)
        D_B_t = to_torch(D_B, device, dtype=torch.float32)
        W_t   = sparse_to_torch(W_A, device, dtype=torch.float32)

        WA_pi = torch.mm(W_t, pi_t)   # sparse @ dense → (n_A, n_B)
        grad  = WA_pi @ D_B_t         # dense @ dense  → (n_A, n_B)
        return (2.0 * grad).cpu().numpy().astype(np.float64)

    # ── CPU path ───────────────────────────────────────────────────────────
    WA_pi = W_A @ pi          # sparse @ dense
    grad  = WA_pi @ D_B       # dense  @ dense
    return (2.0 * grad).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Overlap fraction estimator (trivial — no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_overlap_fraction(
    pi: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Estimate the overlap fraction s = Σ π_ij from the current transport plan.

    s ≈ 1 means nearly full overlap; s ≈ 0.4 means ~40% of cells matched.

    Parameters
    ----------
    pi : (n_A, n_B) float — transport plan.
    a  : (n_A,) float     — source marginal.
    b  : (n_B,) float     — target marginal.

    Returns
    -------
    s : float ∈ (0, 1].
    """
    return float(np.clip(pi.sum(), 1e-6, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: augment_fgw_gradient
# ─────────────────────────────────────────────────────────────────────────────

def augment_fgw_gradient(
    pi: np.ndarray,
    W_A: sp.csr_matrix,
    D_B: np.ndarray,
    lambda_spatial: float,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Return λ_spatial · ∂R_spatial/∂π, ready to add to the FGW gradient.

    Called during the contiguity post-refinement step in ``pairwise_align_se``.
    Returns a zero array when ``lambda_spatial == 0.0`` (no computation done).

    Parameters
    ----------
    pi             : (n_A, n_B) float — current transport plan.
    W_A            : (n_A, n_A) scipy CSR — spatial affinity.
    D_B            : (n_B, n_B) float    — pairwise distances in B.
    lambda_spatial : float — contiguity regularisation weight.
    use_gpu        : bool, default False.

    Returns
    -------
    (n_A, n_B) float64 numpy array.
    """
    if lambda_spatial == 0.0:
        return np.zeros_like(pi, dtype=np.float64)
    return lambda_spatial * contiguity_gradient(pi, W_A, D_B, use_gpu=use_gpu)