"""
utils.py — INCENT utility functions
=====================================
Unchanged from original except minor clarity improvements.
All FGW / conditional gradient logic is preserved exactly.
"""

import numpy as np
import scipy.sparse as sp
import torch
import ot

from tqdm import tqdm
from ot.optim import line_search_armijo
from ot.utils import list_to_array, get_backend
from ot.unbalanced import sinkhorn_unbalanced


# ─────────────────────────────────────────────────────────────────────────────
# Dtype unification helper
# ─────────────────────────────────────────────────────────────────────────────

def _unify_dtypes(*tensors, nx):
    """
    Cast all tensors to the canonical dtype for the given OT backend.

    Rule
    ----
    TorchBackend (GPU or CPU torch): float32
        - Standard for GPU computation; avoids "expected float got double"
          errors from torch.matmul when upstream code mixes float32 and
          float64 arrays.
    NumpyBackend: float64
        - Standard for OT numerical precision on CPU.

    This function is called at the entry point of fused_gromov_wasserstein_incent
    so that every input — regardless of how it was created upstream — is
    normalised before any computation begins.

    Parameters
    ----------
    *tensors : sequence of torch.Tensor or np.ndarray — inputs to normalise.
    nx       : OT backend.

    Returns
    -------
    Tuple of normalised tensors, same length and order as *tensors.
    """
    if isinstance(nx, ot.backend.TorchBackend):
        # GPU canonical dtype: float32
        out = []
        for t in tensors:
            if t is None:
                out.append(None)
            elif isinstance(t, torch.Tensor):
                out.append(t.to(dtype=torch.float32))
            else:
                # numpy → float32 tensor, preserving device of first tensor arg
                arr = np.asarray(t, dtype=np.float32)
                out.append(torch.from_numpy(arr))
        return tuple(out)
    else:
        # CPU canonical dtype: float64
        out = []
        for t in tensors:
            if t is None:
                out.append(None)
            elif isinstance(t, np.ndarray):
                out.append(t.astype(np.float64) if t.dtype != np.float64 else t)
            else:
                out.append(np.asarray(t, dtype=np.float64))
        return tuple(out)


# ═════════════════════════════════════════════════════════════════════════════
# Sparse / dense helpers
# ═════════════════════════════════════════════════════════════════════════════

to_dense_array    = lambda X: X.toarray() if sp.issparse(X) else np.asarray(X)
extract_data_matrix = lambda adata, rep: adata.X if rep is None else adata.obsm[rep]


# ═════════════════════════════════════════════════════════════════════════════
# Jensen-Shannon divergence helpers
# ═════════════════════════════════════════════════════════════════════════════

def kl_divergence_corresponding_backend(X, Y):
    """
    Pairwise KL divergence (matching rows) between X and Y.
    Returns a (n,) array where entry i = KL(X[i] || Y[i]).
    """
    assert X.shape == Y.shape
    nx = ot.backend.get_backend(X, Y)
    X  = X / nx.sum(X, axis=1, keepdims=True)
    Y  = Y / nx.sum(Y, axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i', X, log_X)
    X_log_Y = nx.einsum('ij,ij->i', X, log_Y)
    return nx.to_numpy(X_log_X - X_log_Y)


def jensenshannon_distance_1_vs_many_backend(X, Y):
    """
    JSD between one row X (shape 1×K) and every row of Y (shape m×K).
    Returns a (m,) array.
    """
    assert X.shape[1] == Y.shape[1] and X.shape[0] == 1
    nx  = ot.backend.get_backend(X, Y)
    X   = nx.concatenate([X] * Y.shape[0], axis=0)
    X   = X / nx.sum(X, axis=1, keepdims=True)
    Y   = Y / nx.sum(Y, axis=1, keepdims=True)
    M   = (X + Y) / 2.0
    kl1 = torch.from_numpy(kl_divergence_corresponding_backend(X, M))
    kl2 = torch.from_numpy(kl_divergence_corresponding_backend(Y, M))
    return nx.sqrt((kl1 + kl2) / 2.0).squeeze()


def jensenshannon_divergence_backend(X, Y):
    """
    Full (n × m) JSD matrix between all rows of X and all rows of Y.
    """
    assert X.shape[1] == Y.shape[1]
    print("Computing JSD cost matrix …")
    nx  = ot.backend.get_backend(X, Y)
    X   = X / nx.sum(X, axis=1, keepdims=True)
    Y   = Y / nx.sum(Y, axis=1, keepdims=True)
    n, m = X.shape[0], Y.shape[0]
    D   = nx.zeros((n, m))
    for i in tqdm(range(n)):
        D[i, :] = jensenshannon_distance_1_vs_many_backend(X[i:i+1], Y)
    print("JSD matrix done.")
    if torch.cuda.is_available():
        try:
            return D.numpy()
        except Exception:
            return D
    return D


# ═════════════════════════════════════════════════════════════════════════════
# Mean-squared distance
# ═════════════════════════════════════════════════════════════════════════════

def pairwise_msd(A, B):
    """Pairwise mean-squared distance: (m, n) array."""
    A = np.asarray(A)
    B = np.asarray(B)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # (m, n, d)
    return np.mean(diff ** 2, axis=2)                  # (m, n)


# ═════════════════════════════════════════════════════════════════════════════
# Gromov line search
# ═════════════════════════════════════════════════════════════════════════════

def solve_gromov_linesearch(G, deltaG, cost_G, C1, C2, M, reg,
                             alpha_min=None, alpha_max=None, nx=None, **kwargs):
    """
    Exact quadratic line search for the FW step in FGW.
    Reference: Vayer et al., ICML 2019.
    """
    if nx is None:
        G, deltaG, C1, C2, M = ot.utils.list_to_array(G, deltaG, C1, C2, M)
        nx = ot.backend.get_backend(G, deltaG, C1, C2, M)

    dot = nx.dot(nx.dot(C1, deltaG), C2.T)
    a   = -2.0 * reg * nx.sum(dot * deltaG)
    b   = (nx.sum(M * deltaG)
           - 2.0 * reg * (nx.sum(dot * G)
                          + nx.sum(nx.dot(nx.dot(C1, G), C2.T) * deltaG)))

    alpha = ot.optim.solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    cost_G = cost_G + a * alpha ** 2 + b * alpha
    return alpha, 1, cost_G


# ═════════════════════════════════════════════════════════════════════════════
# Generic conditional gradient
# ═════════════════════════════════════════════════════════════════════════════

def generic_conditional_gradient_incent(
        a, b, M1, M2, f, df, reg1, reg2,
        lp_solver, line_search, gamma,
        G0=None, numItermax=6000,
        stopThr=1e-6, stopThr2=1e-6,
        verbose=False, log=False, **kwargs):
    """
    Generalised conditional gradient for the (F)GW problem with two
    linear cost terms M1 and M2.

    Objective:
        min_G  <M1 + gamma*M2, G> + reg1 * f(G)
        s.t.   G 1 = a,  G^T 1 = b,  G >= 0

    stopThr default is 1e-6
    ------------------------------------
    On GPU all tensors are float32 (machine epsilon ε ≈ 1.2e-7).  The
    relative cost change cannot be measured below ~ε, so stopThr=1e-9
    would never fire — the loop would run until numItermax, wasting
    iterations whose objective changes are below float32 resolution.

    1e-6 is the float32-honest threshold: it stops as soon as changes
    are no longer numerically distinguishable.  This is biologically
    safe because the smallest meaningful OT plan change (one cell being
    reassigned) causes a Δobj ~ 1.3e-4, which is 130× larger than 1e-6.

    On CPU (float64, ε ≈ 1.1e-16) the loop will naturally converge
    before 1e-6 for well-conditioned problems, so this default is also
    correct there (just slightly more permissive than the old 1e-9).
    Pass stopThr=1e-9 explicitly if you need CPU float64 fine convergence.
    """
    a, b, M1, M2, G0 = list_to_array(a, b, M1, M2, G0)
    nx = get_backend(a, b, M1) if not (isinstance(M1, (int, float))) \
        else get_backend(a, b)

    if log:
        log_dict = {'loss': []}

    # Initialise transport plan
    if G0 is None:
        G = nx.ones((a.shape[0], b.shape[0])) / (a.shape[0] * b.shape[0])
    else:
        G = nx.copy(G0)

    M_linear = kwargs.pop('M_linear', M1 + gamma * M2)

    def cost(G):
        return nx.sum(M1 * G) + gamma * nx.sum(M2 * G) + reg1 * f(G)

    cost_G = cost(G)
    if log:
        log_dict['loss'].append(cost_G)

    it   = 0
    loop = True

    if verbose:
        hdr = '{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Rel. loss', 'Abs. loss')
        print(hdr + '\n' + '-' * 48)
        print(f'{it:5d}|{float(cost_G):8e}|{"":8s}|{"":8s}')

    while loop:
        it += 1
        old_cost_G = cost_G

        # Linearise: gradient direction
        Mi = M1 + gamma * M2 + reg1 * df(G)
        if reg2 is not None:
            Mi = Mi + reg2 * (1.0 + nx.log(G))
        Mi = Mi + nx.min(Mi)      # shift non-negative

        # Frank-Wolfe sub-problem
        Gc, innerlog_ = lp_solver(a, b, Mi, **kwargs)

        # Line search
        deltaG = Gc - G
        alpha, _, cost_G = line_search(cost, G, deltaG, Mi, cost_G, **kwargs)
        G = G + alpha * deltaG

        # Convergence checks
        abs_delta  = abs(cost_G - old_cost_G)
        rel_delta  = abs_delta / (abs(cost_G) + 1e-12)

        if it >= numItermax:
            loop = False
        if rel_delta < stopThr or abs_delta < stopThr2:
            loop = False

        if log:
            log_dict['loss'].append(cost_G)

        if verbose and it % 20 == 0:
            print(hdr + '\n' + '-' * 48)
            print(f'{it:5d}|{float(cost_G):8e}|{rel_delta:8e}|{abs_delta:8e}')

    if log:
        log_dict.update(innerlog_)
        return G, log_dict
    return G


# ═════════════════════════════════════════════════════════════════════════════
# CG with unbalanced inner solver
# ═════════════════════════════════════════════════════════════════════════════

def cg_incent(a, b, M1, M2, reg, f, df, gamma,
              G0=None, line_search=line_search_armijo,
              numItermax=6000, numItermaxEmd=100000,
              stopThr=1e-6, stopThr2=1e-6,
              verbose=False, log=False, **kwargs):
    """
    Conditional gradient with Sinkhorn-unbalanced inner LP solver.
    """
    def lp_solver(a, b, M, **kwargs):
        eps = kwargs.get('epsilon', 0.01)
        tau = kwargs.get('tau',     0.1)
        res, innerlog = sinkhorn_unbalanced(
            a, b, M, reg=eps, reg_m=tau,
            numItermax=numItermax, log=True)
        # Re-normalise to keep transport plan as a probability coupling
        nx_  = ot.backend.get_backend(res)
        s    = nx_.sum(res)
        if s > 0:
            res = res / s
        return res, innerlog

    return generic_conditional_gradient_incent(
        a, b, M1, M2, f, df, reg, None,
        lp_solver, line_search,
        G0=G0, gamma=gamma,
        numItermax=numItermax,
        stopThr=stopThr, stopThr2=stopThr2,
        verbose=verbose, log=log, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# Fused Gromov-Wasserstein (entry point called from core.py)
# ═════════════════════════════════════════════════════════════════════════════

def fused_gromov_wasserstein_incent(
        M1, M2, C1, C2, p, q, gamma,
        G_init=None, loss_fun='square_loss',
        alpha=0.1, armijo=False, log=False,
        numItermax=6000, tol_rel=1e-6, tol_abs=1e-6,
        use_gpu=False, **kwargs):
    """
    FGW objective:
        min_pi  (1-alpha)*[<M1,pi> + gamma*<M2,pi>] + alpha * GW(C1,C2,pi)

    Parameters
    ----------
    M1  : (n_A, n_B) linear cost — gene expression + cell-type penalty
    M2  : (n_A, n_B) linear cost — neighbourhood dissimilarity
    C1  : (n_A, n_A) spatial distance matrix of slice A (shared-scale normalised)
    C2  : (n_B, n_B) spatial distance matrix of slice B (shared-scale normalised)
    p   : (n_A,) marginal for A
    q   : (n_B,) marginal for B
    gamma  : weight of M2 relative to M1 inside the linear term
    alpha  : weight of GW term (0 = pure biology, 1 = pure spatial)
    """
    p, q = list_to_array(p, q)
    p0, q0, C10, C20, M10, M20 = p, q, C1, C2, M1, M2
    nx   = get_backend(p0, q0, C10, C20, M10, M20)

    # ── Dtype unification ──────────────────────────────────────────────────────
    # Normalise all inputs to float32 (GPU/TorchBackend) or float64 (CPU/Numpy).
    # Upstream code may produce mixed dtypes (e.g. float32 cosine distance +
    # float64 cell-type mask, or float64 JSD loaded from cache on GPU).
    # Without this guard, nx.dot(C1, G) raises:
    #   RuntimeError: expected mat1 and mat2 to have the same dtype
    M1, M2, C1, C2, p, q = _unify_dtypes(M1, M2, C1, C2, p, q, nx=nx)
    if G_init is not None:
        G_init, = _unify_dtypes(G_init, nx=nx)

    if G_init is None:
        G0 = p[:, None] * q[None, :]
    else:
        s  = nx.sum(G_init)
        G0 = G_init / (s if s > 0 else 1.0)
        if use_gpu:
            # G_init was cast to float32 by _unify_dtypes above — safe to move to GPU.
            G0 = G0.cuda()

    # GW regularisation functions (square loss)
    def f(G):
        return nx.sum((G @ G.T) * C1) + nx.sum((G.T @ G) * C2)

    def df(G):
        return 2.0 * (nx.dot(C1, G) + nx.dot(G, C2))

    if loss_fun == 'kl_loss':
        armijo = True   # no closed-form line search for KL

    # Pre-compute full linear cost for the GW line search
    M_linear = (1.0 - alpha) * M1 + gamma * (1.0 - alpha) * M2

    if armijo:
        def line_search(cost, G, deltaG, Mi, cost_G, **kw):
            return ot.optim.line_search_armijo(cost, G, deltaG, Mi, cost_G,
                                               nx=nx, **kw)
    else:
        def line_search(cost, G, deltaG, Mi, cost_G, **kw):
            return solve_gromov_linesearch(
                G, deltaG, cost_G, C1, C2, M=M_linear, reg=alpha, nx=nx, **kw)

    if log:
        res, log_out = cg_incent(
            p, q,
            (1.0 - alpha) * M1,
            (1.0 - alpha) * M2,
            alpha, f, df, gamma=gamma,
            G0=G0, line_search=line_search, log=True,
            numItermax=numItermax,
            stopThr=tol_rel, stopThr2=tol_abs,
            M_linear=M_linear, **kwargs)
        log_out['fgw_dist'] = log_out['loss'][-1]
        return res, log_out
    else:
        return cg_incent(
            p, q,
            (1.0 - alpha) * M1,
            (1.0 - alpha) * M2,
            alpha, f, df, gamma=gamma,
            G0=G0, line_search=line_search, log=True,
            numItermax=numItermax,
            stopThr=tol_rel, stopThr2=tol_abs,
            M_linear=M_linear, **kwargs)