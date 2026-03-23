"""
_gpu.py — Shared GPU utility helpers for INCENT-SE
====================================================
All GPU-related logic lives here.  No other module calls
``torch.cuda.is_available()`` directly — they import from this file instead.

Three responsibilities
-----------------------
1. ``resolve_device(use_gpu)``   — decide 'cuda' or 'cpu' once.
2. ``to_torch(x, device, ...)``  — numpy/scipy → torch tensor, on device.
3. ``to_numpy(x)``               — torch tensor → numpy float64.
4. ``sparse_to_torch(W, device)``— scipy CSR → torch sparse CSR, on device.

Why a dedicated module?
-----------------------
Without this, every module that benefits from GPU would need its own
``import torch; if torch.cuda.is_available(): ...`` block.  Keeping that
check in one place makes it trivial to add backends (e.g. ROCm, MPS) later.
"""

import numpy as np


def resolve_device(use_gpu: bool) -> str:
    """
    Return the torch device string to use for computation.

    Parameters
    ----------
    use_gpu : bool — whether the caller requested GPU acceleration.

    Returns
    -------
    'cuda' if use_gpu=True and a CUDA GPU is available, else 'cpu'.

    Notes
    -----
    If torch is not installed this function always returns 'cpu'.
    The caller does not need to handle ImportError separately.
    """
    if not use_gpu:
        return 'cpu'
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def to_torch(x, device: str, dtype=None):
    """
    Convert a numpy array or scipy sparse matrix to a torch tensor.

    Parameters
    ----------
    x      : np.ndarray | scipy sparse | torch.Tensor — input array.
    device : str — 'cuda' or 'cpu'.
    dtype  : torch.dtype or None — if None, preserves the input dtype
             (float32 stays float32, float64 stays float64).

    Returns
    -------
    torch.Tensor on ``device``.
    """
    import torch
    import scipy.sparse as sp

    if sp.issparse(x):
        x = x.toarray()                   # CSR → dense ndarray

    if isinstance(x, torch.Tensor):
        t = x.to(device=device)
        return t.to(dtype=dtype) if dtype is not None else t

    arr = np.asarray(x)
    t   = torch.from_numpy(np.ascontiguousarray(arr))
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device)


def to_numpy(x) -> np.ndarray:
    """
    Convert a torch tensor (or numpy array) to a float64 numpy array.

    Always returns a CPU numpy array regardless of which device ``x`` lives on.
    Falls back to numpy if torch is not installed or the object is not a Tensor.
    """
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64)
    except (ImportError, AttributeError):
        pass
    return np.asarray(x, dtype=np.float64)


def sparse_to_torch(W_csr, device: str, dtype=None):
    """
    Convert a scipy CSR sparse matrix to a torch sparse CSR tensor on ``device``.

    Used to move the spatial affinity matrix W_A to the GPU so that
    the sparse × dense matrix product W_A @ π can be computed there.

    Parameters
    ----------
    W_csr  : scipy.sparse.csr_matrix — the sparse matrix to convert.
    device : str — 'cuda' or 'cpu'.
    dtype  : torch.dtype or None — defaults to torch.float32.

    Returns
    -------
    torch.sparse_csr_tensor on ``device``.
    """
    import torch
    dtype = dtype or torch.float32

    crow = torch.tensor(W_csr.indptr,  dtype=torch.int64)
    col  = torch.tensor(W_csr.indices, dtype=torch.int64)
    val  = torch.tensor(W_csr.data,    dtype=dtype)
    return torch.sparse_csr_tensor(crow, col, val,
                                   size=tuple(W_csr.shape)).to(device)
