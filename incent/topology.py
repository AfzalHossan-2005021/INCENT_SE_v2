"""
topology.py — Topological Fingerprints for INCENT-SE
=====================================================
Computes a **persistent-homology fingerprint** for every cell.  These
fingerprints are SE(2)-invariant and symmetry-discriminative: bilaterally
symmetric brain regions share the same cell-type composition, but their
multi-scale topological connectivity differs due to real anatomical asymmetries.

What is persistent homology?
----------------------------
Given a set of points, build a graph by adding edges whenever two points are
within distance ε of each other.  As ε grows from 0 to ∞:
  - Connected components (H0) merge (birth/death = ε at merge events).

The *Betti-0 curve* B(ε) counts how many H0 components exist at scale ε — a
compact summary of connectivity across all scales.

For INCENT-SE we compute one Betti-0 curve per cell type within a local
neighbourhood of radius r_max around each cell.  Stacking K curves gives a
K·L-dimensional fingerprint per cell.

GPU acceleration
----------------
The per-cell Betti-0 curve computation uses a sequential Union-Find algorithm
that cannot be parallelised on GPU.  However ``fingerprint_cost`` — which
computes the (n_A × n_B) pairwise fingerprint distance matrix — is a large
matrix multiplication and is accelerated on GPU when ``use_gpu=True``.

Public API
----------
compute_fingerprints(adata, radius, n_bins, ...) → (n, K*L) np.ndarray
fingerprint_cost(fp_A, fp_B, metric, use_gpu)   → (n_A, n_B) np.ndarray
"""

import os
import numpy as np
from typing import Optional
from anndata import AnnData
from ._gpu import resolve_device, to_torch, to_numpy


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Local subgraph
# ─────────────────────────────────────────────────────────────────────────────

def _local_subgraph(
    coords: np.ndarray,
    labels: np.ndarray,
    center_idx: int,
    radius: float,
) -> tuple:
    """
    Return coordinates and labels of all cells within ``radius`` of cell
    ``center_idx`` (including the centre cell itself).

    Parameters
    ----------
    coords     : (n, 2) float — all cell coordinates.
    labels     : (n,) str    — all cell-type labels.
    center_idx : int         — index of the centre cell.
    radius     : float       — neighbourhood radius (same units as coords).

    Returns
    -------
    sub_coords : (m, 2) float — neighbourhood coordinates.
    sub_labels : (m,) str    — neighbourhood cell-type labels.
    """
    dists = np.linalg.norm(coords - coords[center_idx], axis=1)
    mask  = dists <= radius
    return coords[mask], labels[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Betti-0 curve (Union-Find — inherently sequential, CPU only)
# ─────────────────────────────────────────────────────────────────────────────

def _betti0_curve(
    sub_coords: np.ndarray,
    sub_labels: np.ndarray,
    target_type: str,
    epsilon_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the Betti-0 persistence curve for one cell type in a subgraph.

    For each scale ε we count the number of connected components among cells
    of ``target_type`` using a Union-Find sweep over sorted edges.

    Parameters
    ----------
    sub_coords  : (m, 2) float — local neighbourhood coordinates.
    sub_labels  : (m,) str    — local neighbourhood cell-type labels.
    target_type : str         — the cell type to analyse.
    epsilon_grid: (L,) float  — sorted scale values ε₁ < … < ε_L.

    Returns
    -------
    betti_curve : (L,) int32 — connected component count at each scale.
        All-zeros if ``target_type`` is absent from the subgraph.
    """
    mask = sub_labels == target_type
    pts  = sub_coords[mask]
    n    = len(pts)

    if n == 0:
        return np.zeros(len(epsilon_grid), dtype=np.int32)
    if n == 1:
        # One isolated point — always 1 component
        return np.ones(len(epsilon_grid), dtype=np.int32)

    # Pairwise distances within the local type-k point cloud
    diff  = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))

    # Union-Find with path compression and union by rank
    parent = np.arange(n)
    rank   = np.zeros(n, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    # Sort edges by distance once, then sweep ε through the grid
    i_idx, j_idx = np.triu_indices(n, k=1)
    edge_dists   = dists[i_idx, j_idx]
    order        = np.argsort(edge_dists)
    sorted_edges = list(zip(i_idx[order], j_idx[order], edge_dists[order]))

    betti_curve  = np.zeros(len(epsilon_grid), dtype=np.int32)
    edge_ptr     = 0
    n_components = n

    for l, eps in enumerate(epsilon_grid):
        while (edge_ptr < len(sorted_edges)
               and sorted_edges[edge_ptr][2] <= eps):
            ei, ej, _ = sorted_edges[edge_ptr]
            if union(ei, ej):
                n_components -= 1
            edge_ptr += 1
        betti_curve[l] = n_components

    return betti_curve


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Full fingerprint for one cell
# ─────────────────────────────────────────────────────────────────────────────

def _cell_fingerprint(
    coords: np.ndarray,
    labels: np.ndarray,
    center_idx: int,
    cell_types: np.ndarray,
    radius: float,
    epsilon_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the topological fingerprint for a single cell.

    Fingerprint = concatenation of K Betti-0 curves, one per cell type:
        f_i = [ B^(ct_1)(ε₁..ε_L), …, B^(ct_K)(ε₁..ε_L) ]   shape (K·L,)

    The vector is L2-normalised so magnitude does not depend on local density.

    Parameters
    ----------
    coords      : (n, 2) float — all cell coordinates.
    labels      : (n,) str    — all cell-type labels.
    center_idx  : int         — index of the cell to fingerprint.
    cell_types  : (K,) str    — cell types to include.
    radius      : float       — local neighbourhood radius.
    epsilon_grid: (L,) float  — sorted spatial scales.

    Returns
    -------
    fp : (K·L,) float32.
    """
    sub_coords, sub_labels = _local_subgraph(coords, labels, center_idx, radius)

    curves = [_betti0_curve(sub_coords, sub_labels, ct, epsilon_grid).astype(np.float32)
              for ct in cell_types]

    fp   = np.concatenate(curves)
    norm = np.linalg.norm(fp)
    if norm > 1e-10:
        fp /= norm
    return fp


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: compute_fingerprints
# ─────────────────────────────────────────────────────────────────────────────

def compute_fingerprints(
    adata: AnnData,
    radius: float,
    n_bins: int = 16,
    cache_path: Optional[str] = None,
    slice_name: str = "slice",
    overwrite: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute topological fingerprints for every cell in ``adata``.

    Results are cached to disk so they are only computed once per slice.
    The per-cell Betti-0 computation is sequential (Union-Find) and runs on
    CPU.  The resulting fingerprint matrix is used by ``fingerprint_cost``
    which IS GPU-accelerated.

    Parameters
    ----------
    adata      : AnnData — must have ``.obsm['spatial']`` and
                 ``.obs['cell_type_annot']``.
    radius     : float   — neighbourhood radius (same units as spatial coords).
                 Should match the ``radius`` used for JSD in ``pairwise_align``.
    n_bins     : int, default 16
                 Number of scale bins in the Betti-0 curve (L).
                 More bins → richer fingerprint, slower computation.
    cache_path : str or None — directory for .npy cache files.
    slice_name : str — identifier used in the cache filename.
    overwrite  : bool — if True, recompute even if a cache file exists.
    verbose    : bool — show tqdm progress bar.

    Returns
    -------
    fingerprints : (n_cells, K·n_bins) float32 array.
    """
    from tqdm import tqdm

    # ── Check cache ────────────────────────────────────────────────────────
    if cache_path is not None:
        os.makedirs(cache_path, exist_ok=True)
        cache_file = os.path.join(cache_path, f"topo_fp_{slice_name}.npy")
        if os.path.exists(cache_file) and not overwrite:
            if verbose:
                print(f"[topology] Loading cached fingerprints from {cache_file}")
            return np.load(cache_file)

    coords     = adata.obsm['spatial'].astype(np.float64)
    labels     = np.asarray(adata.obs['cell_type_annot'].astype(str))
    cell_types = np.unique(labels)
    n_cells    = len(coords)
    K, L       = len(cell_types), n_bins

    # Scale grid: L evenly-spaced values from 0 to radius
    # (we start at radius/L rather than 0 to avoid the trivial ε=0 case)
    epsilon_grid = np.linspace(radius / L, radius, L)

    if verbose:
        print(f"[topology] {n_cells} cells | K={K} types | L={L} bins | r={radius}")

    fingerprints = np.zeros((n_cells, K * L), dtype=np.float32)

    for i in tqdm(range(n_cells), desc="Topological fingerprints",
                  disable=not verbose):
        fingerprints[i] = _cell_fingerprint(
            coords, labels, i, cell_types, radius, epsilon_grid)

    # ── Cache ──────────────────────────────────────────────────────────────
    if cache_path is not None:
        np.save(cache_file, fingerprints)
        if verbose:
            print(f"[topology] Cached to {cache_file}")

    return fingerprints


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: fingerprint_cost   ← GPU-accelerated
# ─────────────────────────────────────────────────────────────────────────────

def fingerprint_cost(
    fp_A: np.ndarray,
    fp_B: np.ndarray,
    metric: str = 'cosine',
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Compute the pairwise topological dissimilarity matrix M_topo.

    M_topo[i, j] = distance(fingerprint_i^A, fingerprint_j^B)

    This is the third linear cost term in the FGW objective alongside M1
    (gene expression) and M2 (neighbourhood JSD).  It penalises matching
    cells with different multi-scale topological neighbourhoods, which
    distinguishes left-vs-right hemisphere even when composition is identical.

    GPU acceleration
    ----------------
    The core operation is a row-normalised matrix product:
        M_topo = 1 − fp_A_norm @ fp_B_norm.T   (for cosine)
    With n_A = n_B = 15k and D = 320 this is a 15k × 320 × 15k GEMM —
    a substantial GPU win (~10–40× faster than CPU for these sizes).

    Parameters
    ----------
    fp_A    : (n_A, D) float32 — fingerprints from sliceA.
    fp_B    : (n_B, D) float32 — fingerprints from sliceB.
    metric  : 'cosine' (default) or 'euclidean'.
    use_gpu : bool, default False — use CUDA if available.

    Returns
    -------
    M_topo : (n_A, n_B) float32 numpy array.
    """
    device = resolve_device(use_gpu)

    if device == 'cuda':
        import torch
        # Move to GPU as float32 (sufficient precision for a cost matrix)
        A = to_torch(fp_A, device, dtype=torch.float32)
        B = to_torch(fp_B, device, dtype=torch.float32)

        if metric == 'cosine':
            A = A / (A.norm(dim=1, keepdim=True) + 1e-10)
            B = B / (B.norm(dim=1, keepdim=True) + 1e-10)
            M = 1.0 - A @ B.T

        elif metric == 'euclidean':
            sq_A = (A ** 2).sum(dim=1, keepdim=True)   # (n_A, 1)
            sq_B = (B ** 2).sum(dim=1, keepdim=True).T  # (1, n_B)
            M    = torch.clamp(sq_A + sq_B - 2.0 * (A @ B.T), min=0.0).sqrt()

        else:
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")

        return M.cpu().numpy().astype(np.float32)

    # ── CPU path ───────────────────────────────────────────────────────────
    A = fp_A.astype(np.float32)
    B = fp_B.astype(np.float32)

    if metric == 'cosine':
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
        return (1.0 - A @ B.T).astype(np.float32)

    elif metric == 'euclidean':
        sq_A = (A ** 2).sum(axis=1, keepdims=True)
        sq_B = (B ** 2).sum(axis=1, keepdims=True).T
        return np.sqrt(np.maximum(sq_A + sq_B - 2.0 * (A @ B.T), 0.0)).astype(np.float32)

    else:
        raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")