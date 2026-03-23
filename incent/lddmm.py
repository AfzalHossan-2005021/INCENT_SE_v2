"""
lddmm.py — LDDMM Diffeomorphic Deformation for INCENT-SE Stage 5
=================================================================
Recovers the spatial deformation field between two slices from different
developmental timepoints using Large Deformation Diffeomorphic Metric Mapping.

What is LDDMM?
--------------
The diffeomorphism φ is the flow of a stationary velocity field v:
    φ(y) = y + Σ_{t=1}^{T} v(y_t) / T    (Euler integration)
    v(x)  = Σ_j  K(x, c_j) · α_j         (kernel interpolation)

K is a Gaussian RKHS kernel with bandwidth σ_v that controls deformation
smoothness.  α are the momentum vectors at control points c (= cell locations
of sliceB, subsampled to ≤ 2000 for memory efficiency).

The deformation minimises:
    E(φ) = Σ_{i,j} π_{ij} ||φ(y_j) - x_i||²   +   λ_V · ||v||²_V

GPU acceleration
----------------
All matrix-heavy operations use torch tensors on the chosen device:

  _gaussian_kernel   : (n, m) distance matrix + exp — pure matmul, big GPU win.
  velocity_at        : kernel @ alpha — (n×m) @ (m×2).
  apply (Euler loop) : T repeated kernel matmuls.
  _transport_loss    : vectorised O(n_B × n_A) computation on GPU.
  deformed_distances : (n_B × n_B) pairwise distance on GPU.

The ``LDDMMDeformation`` class stores alpha as a torch tensor on the target
device during training.  ``apply`` always returns a CPU numpy array so
downstream code does not need to be GPU-aware.

Public API
----------
LDDMMDeformation          — stores the velocity field; apply/rkhs ops.
estimate_deformation(...)  → LDDMMDeformation
deformed_distances(...)    → (n_B, n_B) np.ndarray
estimate_growth_vector(...)→ (n_B,) np.ndarray
"""

import numpy as np
from typing import Tuple
from ._gpu import resolve_device, to_torch, to_numpy


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian kernel — device-aware (numpy or torch)
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(x, y, sigma: float):
    """
    Evaluate the Gaussian kernel matrix K(x, y).

    K[i,j] = exp( -||x_i - y_j||² / (2σ²) )

    Accepts either numpy arrays (CPU) or torch tensors (GPU).
    Returns the same type as the inputs.

    Parameters
    ----------
    x     : (n, 2) array/tensor — first set of points.
    y     : (m, 2) array/tensor — second set of points.
    sigma : float — kernel bandwidth (spatial scale of deformation).

    Returns
    -------
    K : (n, m) — same type and device as x.
    """
    try:
        import torch
        if isinstance(x, torch.Tensor):
            sq_x = (x ** 2).sum(dim=1, keepdim=True)    # (n, 1)
            sq_y = (y ** 2).sum(dim=1, keepdim=True).T   # (1, m)
            D2   = (sq_x + sq_y - 2.0 * x @ y.T).clamp(min=0.0)
            return torch.exp(-D2 / (2.0 * sigma ** 2))
    except ImportError:
        pass

    # Numpy path
    sq_x = (x ** 2).sum(axis=1, keepdims=True)
    sq_y = (y ** 2).sum(axis=1, keepdims=True).T
    D2   = np.maximum(sq_x + sq_y - 2.0 * (x @ y.T), 0.0)
    return np.exp(-D2 / (2.0 * sigma ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# LDDMMDeformation class — GPU-aware
# ─────────────────────────────────────────────────────────────────────────────

class LDDMMDeformation:
    """
    Parametric LDDMM deformation defined by momentum vectors at control points.

    The velocity field is:
        v(x) = Σ_j  K(x, c_j) · α_j

    and is Euler-integrated to give the diffeomorphism φ.

    Parameters
    ----------
    control_points : (m, 2) float array — positions where momentum is defined.
    sigma_v        : float — Gaussian kernel bandwidth.
    n_steps        : int   — Euler integration steps (more = smoother φ).
    device         : str   — 'cuda' or 'cpu'.  All internal tensors live here.
    """

    def __init__(
        self,
        control_points: np.ndarray,
        sigma_v: float,
        n_steps: int = 5,
        device: str = 'cpu',
    ):
        self.sigma_v  = sigma_v
        self.n_steps  = n_steps
        self.device   = device

        ctrl_np = control_points.astype(np.float64)
        m       = len(ctrl_np)

        if device == 'cuda':
            # Store on GPU for fully-GPU gradient descent loop
            self._ctrl = to_torch(ctrl_np, device)
            try:
                import torch
                self.alpha = torch.zeros((m, 2), dtype=torch.float64, device=device)
            except ImportError:
                self.alpha = np.zeros((m, 2), dtype=np.float64)
        else:
            # CPU: keep everything as numpy — no torch dependency needed
            self._ctrl = ctrl_np
            self.alpha = np.zeros((m, 2), dtype=np.float64)

        # Pre-compute kernel matrix at control points (device-agnostic)
        self._K_cc = _gaussian_kernel(self._ctrl, self._ctrl, sigma_v)

    def velocity_at(self, query_points):
        """
        Evaluate v(x) = K(x, ctrl) · α at query_points.

        Parameters
        ----------
        query_points : (n, 2) tensor/array on self.device.

        Returns
        -------
        v : (n, 2) — same type and device as query_points.
        """
        K = _gaussian_kernel(query_points, self._ctrl, self.sigma_v)
        return K @ self.alpha

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply φ to coordinates via Euler integration.  Always returns numpy.

        Parameters
        ----------
        coords : (n, 2) float — points to deform (any device/dtype).

        Returns
        -------
        deformed : (n, 2) float64 numpy array — φ(coords).
        """
        coords_f = coords.astype(np.float64)
        dt       = 1.0 / self.n_steps

        if self.device == 'cuda':
            y = to_torch(coords_f, self.device)
            for _ in range(self.n_steps):
                y = y + dt * self.velocity_at(y)
            return to_numpy(y)

        # CPU: stay in numpy the whole way
        y = coords_f.copy()
        for _ in range(self.n_steps):
            y = y + dt * self.velocity_at(y)
        return y

    def rkhs_norm_squared(self) -> float:
        """
        ||v||²_V = α^T · K_cc · α  (RKHS regularisation term).

        Returns
        -------
        float.
        """
        try:
            import torch
            if isinstance(self.alpha, torch.Tensor):
                val = (self.alpha * (self._K_cc @ self.alpha)).sum()
                return float(val.item())
        except ImportError:
            pass
        return float(np.einsum('id,ij,jd->', self.alpha, self._K_cc, self.alpha))

    def rkhs_norm_gradient(self):
        """
        ∂||v||²_V / ∂α = 2 · K_cc · α.

        Returns
        -------
        (m, 2) — same type and device as self.alpha.
        """
        return 2.0 * (self._K_cc @ self.alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Transport loss — fully vectorised, GPU-accelerated
# ─────────────────────────────────────────────────────────────────────────────

def _transport_loss(
    phi: LDDMMDeformation,
    pi,          # torch tensor (n_A, n_B) on phi.device
    coords_A,    # torch tensor (n_A, 2)  on phi.device
    coords_B,    # torch tensor (n_B, 2)  on phi.device
) -> Tuple:
    """
    Compute E_transport = Σ_{i,j} π_{ij} ||φ(y_j) - x_i||² and ∂E/∂α.

    Fully vectorised (no Python j-loop).  All tensors on phi.device.

    Vectorised loss formula
    -----------------------
    diff²[i,j] = ||y_j^φ - x_i||²
               = ||y_j^φ||² − 2 y_j^φ · x_i + ||x_i||²
    loss = Σ_{i,j} π_{ij} · diff²[i,j]
         = (π · sq_yφ.T).sum() − 2(π ⊙ (x @ yφ.T).T).sum() + (π · sq_x).sum()

    Re-expressed in matrix form (memory-efficient — only O(n_B + n_A) overhead):
        pi_col = π^T · 1          (n_B,) marginal
        Σ_i π_{ij} x_i = π^T @ x (n_B, 2) weighted centroid
        residual_j = pi_col_j · y_j^φ − (π^T @ x)_j    (n_B, 2)
        loss = sq_yφ · pi_col − 2 · diag(yφ @ (π^T @ x)^T) + sq_x · π^T·1
             = (pi_col * sq_yφ).sum() + sq_x @ pi_col − 2*(y_phi * weighted_xA).sum()

    Gradient ∂E/∂α = 2 · K(y^φ, c)^T · residuals   (m, 2)

    Parameters  (all torch tensors on phi.device)
    ----------
    phi      : LDDMMDeformation.
    pi       : (n_A, n_B) float64.
    coords_A : (n_A, 2)   float64.
    coords_B : (n_B, 2)   float64 — original (pre-deformation) sliceB coords.

    Returns
    -------
    (loss: float,  grad_alpha: (m, 2) tensor on phi.device)
    """
    import torch

    y_phi = phi.apply(to_numpy(coords_B))          # always returns numpy
    y_phi = to_torch(y_phi, phi.device)            # back to device

    pi_col         = pi.sum(dim=0)                 # (n_B,)
    weighted_xA    = pi.T @ coords_A               # (n_B, 2) = Σ_i π_{ij} x_i

    sq_yφ = (y_phi ** 2).sum(dim=1)               # (n_B,)
    sq_xA = (coords_A ** 2).sum(dim=1)            # (n_A,)

    loss = (pi_col * sq_yφ).sum() \
         + (sq_xA @ pi.sum(dim=1)) \
         - 2.0 * (y_phi * weighted_xA).sum()

    # Residuals for gradient: r_j = pi_col_j · y_j^φ − Σ_i π_{ij} x_i
    residuals = pi_col.unsqueeze(1) * y_phi - weighted_xA   # (n_B, 2)

    K_yB_c    = _gaussian_kernel(y_phi, phi._ctrl, phi.sigma_v)  # (n_B, m)
    grad_alpha = 2.0 * K_yB_c.T @ residuals                      # (m, 2)

    return float(loss.item()), grad_alpha


def _transport_loss_numpy(
    phi: LDDMMDeformation,
    pi: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """CPU (numpy) fallback for _transport_loss."""
    y_phi = phi.apply(coords_B)                    # (n_B, 2) numpy

    pi_col      = pi.sum(axis=0)                   # (n_B,)
    weighted_xA = pi.T @ coords_A                  # (n_B, 2)

    sq_yφ = (y_phi ** 2).sum(axis=1)
    sq_xA = (coords_A ** 2).sum(axis=1)

    loss = ((pi_col * sq_yφ).sum()
            + sq_xA @ pi.sum(axis=1)
            - 2.0 * (y_phi * weighted_xA).sum())

    residuals  = pi_col[:, None] * y_phi - weighted_xA   # (n_B, 2)
    K_yB_c    = _gaussian_kernel(y_phi, to_numpy(phi._ctrl), phi.sigma_v)
    grad_alpha = 2.0 * K_yB_c.T @ residuals              # (m, 2)

    return float(loss), grad_alpha


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: estimate_deformation — GPU-accelerated gradient descent
# ─────────────────────────────────────────────────────────────────────────────

def estimate_deformation(
    pi: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    sigma_v: float,
    lambda_v: float = 1.0,
    n_steps: int = 5,
    lr: float = 0.01,
    n_iter: int = 100,
    tol: float = 1e-6,
    use_gpu: bool = False,
    verbose: bool = False,
) -> LDDMMDeformation:
    """
    Estimate the LDDMM diffeomorphism φ mapping sliceB coordinates onto sliceA.

    Given the current OT plan π, minimises:
        E(φ) = Σ_{i,j} π_{ij} ||φ(y_j) - x_i||²   +   λ_V · ||v||²_V

    The first term pulls deformed sliceB cells towards their matched sliceA
    cells.  The RKHS norm ensures the deformation is smooth.

    GPU acceleration
    ----------------
    All internal tensors (alpha, kernel matrices, gradients) live on the GPU
    when ``use_gpu=True`` and CUDA is available.  The gradient descent loop
    runs entirely on GPU — only the final deformation object is returned to
    CPU (via ``phi.apply()`` which always returns numpy).

    Parameters
    ----------
    pi       : (n_A, n_B) float — OT plan from the FGW step.
    coords_A : (n_A, 2)   float — cell coordinates in sliceA.
    coords_B : (n_B, 2)   float — cell coordinates in sliceB.
    sigma_v  : float — RKHS kernel bandwidth.  For MERFISH: 200–1000 μm.
    lambda_v : float, default 1.0 — RKHS regularisation weight.
    n_steps  : int, default 5 — Euler integration steps per forward pass.
    lr       : float, default 0.01 — gradient descent learning rate.
    n_iter   : int, default 100 — maximum gradient steps.
    tol      : float, default 1e-6 — relative loss convergence threshold.
    use_gpu  : bool, default False.
    verbose  : bool, default False.

    Returns
    -------
    phi : LDDMMDeformation — use ``phi.apply(coords_B)`` for deformed coords.
    """
    device   = resolve_device(use_gpu)
    coords_A = coords_A.astype(np.float64)
    coords_B = coords_B.astype(np.float64)

    # ── Subsample control points (kernel matrix is O(m²)) ─────────────────
    n_B, max_ctrl = len(coords_B), 2000
    if n_B > max_ctrl:
        idx_ctrl = np.random.choice(n_B, max_ctrl, replace=False)
        ctrl_pts = coords_B[idx_ctrl]
        if verbose:
            print(f"[lddmm] Subsampled {max_ctrl} control points from {n_B} cells")
    else:
        ctrl_pts = coords_B

    phi = LDDMMDeformation(ctrl_pts, sigma_v=sigma_v, n_steps=n_steps, device=device)

    # ── Move constant tensors to device once ──────────────────────────────
    use_torch = (device == 'cuda')
    if use_torch:
        import torch
        pi_d  = to_torch(pi,       device, dtype=torch.float64)
        cA_d  = to_torch(coords_A, device, dtype=torch.float64)
        cB_d  = to_torch(coords_B, device, dtype=torch.float64)
    else:
        pi_d, cA_d, cB_d = pi, coords_A, coords_B

    prev_loss = np.inf

    for it in range(n_iter):
        if use_torch:
            E_t,  grad_t = _transport_loss(phi, pi_d, cA_d, cB_d)
        else:
            E_t,  grad_t = _transport_loss_numpy(phi, pi_d, cA_d, cB_d)

        E_v    = phi.rkhs_norm_squared()
        grad_v = phi.rkhs_norm_gradient()

        total_loss = E_t + lambda_v * E_v
        total_grad = grad_t + lambda_v * grad_v

        if verbose and it % 10 == 0:
            print(f"[lddmm] it={it:4d}  E_t={E_t:.4f}  "
                  f"E_v={E_v:.4f}  total={total_loss:.4f}")

        # ── Convergence check ─────────────────────────────────────────────
        rel_change = abs(total_loss - prev_loss) / (abs(prev_loss) + 1e-12)
        if rel_change < tol and it > 5:
            if verbose:
                print(f"[lddmm] Converged at iteration {it}")
            break
        prev_loss = total_loss

        phi.alpha = phi.alpha - lr * total_grad

    return phi


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: deformed_distances — GPU-accelerated pairwise distance
# ─────────────────────────────────────────────────────────────────────────────

def deformed_distances(
    coords_B: np.ndarray,
    phi: LDDMMDeformation,
    normalise: bool = True,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances of the deformed sliceB coordinates.

    Used to update D_B in the BCD loop: the GW term should use D_B(φ) =
    pairwise distances of φ(coords_B) rather than the original D_B.

    GPU path
    --------
    The deformed coordinates y^φ = phi.apply(coords_B) are computed by
    the LDDMMDeformation object (already GPU-aware).  The (n_B × n_B)
    pairwise distance matrix is then computed via the identity:

        ||y_i - y_j||² = ||y_i||² + ||y_j||² - 2 y_i·y_j

    as a single matrix product on GPU — far faster than the naive (n, n, 2)
    broadcast for large n.

    Parameters
    ----------
    coords_B  : (n_B, 2) float — original sliceB coordinates.
    phi       : LDDMMDeformation — estimated deformation.
    normalise : bool, default True — divide by max(D_B(φ)).
    use_gpu   : bool, default False.

    Returns
    -------
    D_B_deformed : (n_B, n_B) float64 numpy array.
    """
    device = resolve_device(use_gpu)
    y_phi  = phi.apply(coords_B)   # (n_B, 2) numpy — always returned by apply()

    if device == 'cuda':
        import torch
        y   = to_torch(y_phi, device, dtype=torch.float32)
        sq  = (y ** 2).sum(dim=1, keepdim=True)            # (n_B, 1)
        D2  = (sq + sq.T - 2.0 * y @ y.T).clamp(min=0.0)  # (n_B, n_B)
        D   = D2.sqrt()
        if normalise:
            m = D.max()
            if m > 1e-12:
                D = D / m
        return D.cpu().numpy().astype(np.float64)

    # ── CPU path ───────────────────────────────────────────────────────────
    sq  = (y_phi ** 2).sum(axis=1, keepdims=True)
    D2  = np.maximum(sq + sq.T - 2.0 * (y_phi @ y_phi.T), 0.0)
    D   = np.sqrt(D2)
    if normalise:
        m = D.max()
        if m > 1e-12:
            D /= m
    return D.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: estimate_growth_vector (trivial — no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_growth_vector(
    pi: np.ndarray,
    b: np.ndarray,
    kappa: float = 0.1,
) -> np.ndarray:
    """
    Estimate the per-cell growth vector ξ from the transport plan.

    ξ[j] > 1 → cell j in sliceB proliferates / expands.
    ξ[j] < 1 → cell j contracts / is dying.
    ξ[j] ≈ 1 → stable population.

    Estimation: ξ_j = (π^T·1)[j] / b[j], regularised towards 1 by κ.

    Parameters
    ----------
    pi    : (n_A, n_B) float — transport plan.
    b     : (n_B,) float    — target marginal.
    kappa : float, default 0.1 — prior strength (larger → ξ closer to 1).

    Returns
    -------
    xi : (n_B,) float64.
    """
    pi_col = pi.sum(axis=0)
    raw    = pi_col / (b + 1e-12)
    xi     = (raw + kappa) / (1.0 + kappa)
    return xi.astype(np.float64)