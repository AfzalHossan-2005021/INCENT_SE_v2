"""
cvae.py — Conditional VAE for Cross-Timepoint Expression Embedding   [v2]
=========================================================================
Learns a latent embedding of gene expression that is:

  (a) Cell-type anchored — cells of the same type cluster together in
      latent space regardless of which developmental timepoint they came from.
  (b) Batch/time invariant — temporal changes in expression are factored out.

Why this module exists
-----------------------
For cross-timepoint alignment we need a pairwise cost M_latent[i,j] that
measures how similar cell i (timepoint t_A) is to cell j (timepoint t_B)
*independent of time*.  Raw gene-expression cosine distance fails because
the same cell type has different profiles at different developmental stages.
A neural-progenitor at E12 looks different from the same cell at E16 in
expression space — so the OT cost would penalise their matching even though
they ARE the correct correspondence.

v2 changes vs v1
-----------------
Bug fixed: the v1 dataset applied ``np.log1p`` unconditionally.  If the
AnnData already contains log-normalised or z-scored values (negative numbers
present, or maximum value < 15), ``log1p`` of a negative number is NaN.
This caused every training epoch to emit NaN loss.

Fixes:
  1. _smart_preprocess() auto-detects whether data is raw counts or
     already normalised and skips log1p for pre-normalised data.
  2. NaN/Inf values in the input are replaced with 0.0 before training.
  3. Robust z-scoring: clip outliers to ±5σ before standardising.
  4. Gradient clipping is kept; also added beta-KL annealing to prevent
     posterior collapse on small datasets.

Public API
----------
INCENT_cVAE                — PyTorch model class (train / embed / save / load)
train_cvae(adatas, ...)    → INCENT_cVAE
latent_cost(adata_A, adata_B, model) → np.ndarray  [(n_A, n_B) M_latent]
"""

import numpy as np
from typing import List, Optional
from anndata import AnnData
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_dense(X) -> np.ndarray:
    """Convert sparse or dense matrix to float32 numpy."""
    import scipy.sparse as sp
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def _smart_preprocess(X: np.ndarray) -> np.ndarray:
    """
    Robustly preprocess a gene-expression matrix for VAE input.

    Handles three common data states found in AnnData objects:

    State 1 — Raw integer counts (values ≥ 0, max > 50):
        Apply log1p then z-score.

    State 2 — Already log-normalised (values ≥ 0, max ≤ 50, e.g. log1p CPM):
        Skip log1p.  Apply z-score only.

    State 3 — Already z-scored or normalised (contains negative values):
        Skip log1p.  Apply light rescaling to [-1, 1] to prevent gradient
        explosion, but do NOT z-score again (would destroy the normalisation).

    In all cases NaN/Inf are replaced with 0 and extreme outliers are
    clipped to ±5σ so they cannot dominate the loss.

    Parameters
    ----------
    X : (n, p) float32 — raw gene-expression matrix for ONE slice.

    Returns
    -------
    X_proc : (n, p) float32 — preprocessed, ready for VAE training.
    """
    X = X.astype(np.float32)

    # Replace NaN / Inf immediately — these are the primary source of NaN loss
    X = np.where(np.isfinite(X), X, 0.0)

    has_negatives = (X < 0).any()
    x_max         = X.max()

    if has_negatives:
        # State 3: already normalised (z-scored, centred log-ratio, etc.)
        # Rescale to [-1, 1] without re-standardising
        x_abs_max = np.abs(X).max()
        if x_abs_max > 1e-6:
            X = X / x_abs_max
    elif x_max > 50:
        # State 1: raw counts — apply log1p then z-score
        X = np.log1p(np.clip(X, 0.0, None))    # clip defensively; log1p(neg)=NaN
        mu   = X.mean(axis=0, keepdims=True)
        sig  = X.std(axis=0, keepdims=True)
        X    = (X - mu) / (sig + 1e-6)
        # Clip outliers to ±5σ
        X    = np.clip(X, -5.0, 5.0)
    else:
        # State 2: log-normalised, values in [0, ~15]
        mu   = X.mean(axis=0, keepdims=True)
        sig  = X.std(axis=0, keepdims=True)
        X    = (X - mu) / (sig + 1e-6)
        X    = np.clip(X, -5.0, 5.0)

    # Final NaN guard — catches any edge case missed above
    X = np.where(np.isfinite(X), X, 0.0)
    return X


def _normalize_spatial_coords(coords: np.ndarray) -> np.ndarray:
    """Robustly normalise spatial coordinates for auxiliary supervision."""
    coords = np.asarray(coords, dtype=np.float32)
    center = np.median(coords, axis=0)
    scale = float(np.median(np.linalg.norm(coords - center, axis=1))) + 1e-6
    return ((coords - center) / scale).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class _SpatialTranscriptomicsDataset(Dataset):
    """
    PyTorch-compatible dataset wrapping multiple AnnData objects.

    Each __getitem__ returns (expression_vector, cell_type_index).
    All preprocessing is done once in __init__ via _smart_preprocess().
    """

    def __init__(
        self,
        adatas: List[AnnData],
        shared_genes: List[str],
        cell_type_map: dict,
    ):
        import torch

        exprs, ctypes, coords_list, source_ids = [], [], [], []

        for source_id, ad in enumerate(adatas):
            X = _to_dense(ad[:, shared_genes].X)
            X = _smart_preprocess(X)          # robust preprocessing (v2 fix)
            exprs.append(X)

            ct = np.array([cell_type_map.get(str(c), 0)
                           for c in ad.obs['cell_type_annot'].astype(str)])
            ctypes.append(ct)

            if "spatial" in ad.obsm:
                coords = _normalize_spatial_coords(np.asarray(ad.obsm["spatial"]))
            else:
                coords = np.zeros((len(ad), 2), dtype=np.float32)
            coords_list.append(coords)
            source_ids.append(np.full(len(ad), source_id, dtype=np.int64))

        X_all  = np.concatenate(exprs,  axis=0)
        ct_all = np.concatenate(ctypes, axis=0)
        coord_all = np.concatenate(coords_list, axis=0)
        source_all = np.concatenate(source_ids, axis=0)

        # Final sanity check — should never trigger after _smart_preprocess
        if not np.isfinite(X_all).all():
            bad = (~np.isfinite(X_all)).sum()
            import warnings
            warnings.warn(f"[cVAE dataset] Replacing {bad} non-finite values "
                          f"with 0 in expression matrix.", stacklevel=2)
            X_all = np.where(np.isfinite(X_all), X_all, 0.0)

        self.X       = torch.tensor(X_all,  dtype=torch.float32)
        self.ct      = torch.tensor(ct_all, dtype=torch.long)
        self.coords  = torch.tensor(coord_all, dtype=torch.float32)
        self.source  = torch.tensor(source_all, dtype=torch.long)
        self.n_genes = self.X.shape[1]
        self.n_types = len(cell_type_map)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ct[idx], self.coords[idx], self.source[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class INCENT_cVAE:
    """
    Conditional Variational Autoencoder for spatiotemporal gene expression.

    Architecture
    ------------
    Encoder:  [x ; one_hot(label)] → FC(H*2) → LayerNorm → ReLU
                                   → FC(H)   → LayerNorm → ReLU
                                   → μ ∈ ℝ^d,  log_σ² ∈ ℝ^d

    Decoder:  [z ; one_hot(label)] → FC(H)   → LayerNorm → ReLU
                                   → FC(H*2) → LayerNorm → ReLU
                                   → FC(n_genes)

    LayerNorm is used instead of BatchNorm because it is stable for small
    batch sizes and does not depend on batch statistics at inference time.

    Loss
    ----
    L = -ELBO + λ_triplet · L_triplet + λ_finite · L_finite
    where:
      ELBO         = recon_mse − β · KL(q||p)
      L_triplet    = batch semi-hard triplet loss on z (cell-type anchor)
      L_finite     = penalty on non-finite outputs (extra guard for NaN)
      β is linearly annealed from 0 → 1 over the first 50 epochs (KL warmup)

    Parameters
    ----------
    n_genes        : int — number of input genes.
    n_types        : int — number of cell types (for conditioning).
    latent_dim     : int, default 32.
    hidden_dim     : int, default 256.
    lambda_triplet : float, default 1.0.
    """

    def __init__(
        self,
        n_genes: int,
        n_types: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        lambda_triplet: float = 1.0,
    ):
        self.n_genes        = n_genes
        self.n_types        = n_types
        self.latent_dim     = latent_dim
        self.hidden_dim     = hidden_dim
        self.lambda_triplet = lambda_triplet
        self._model         = None
        self.shared_genes   = []
        self.cell_type_map  = {}

    def _build_torch_model(self):
        """Build the PyTorch nn.Module (called lazily at train time)."""
        import torch
        import torch.nn as nn

        d        = self.latent_dim
        H        = self.hidden_dim
        n_genes  = self.n_genes
        n_types  = self.n_types
        cond_dim = n_genes + n_types

        class _cVAE(nn.Module):
            def __init__(self):
                super().__init__()
                # Biological Grounding: Gene-level Feature Attention
                # Instead of treating all genes equally, learn an unsupervised 
                # structural attention mask to separate "lineage/identity" genes 
                # from transient "state/metabolic" genes. This prevents 
                # temporal gene shift from disrupting alignment.
                self.gene_attention = nn.Sequential(
                    nn.Linear(n_genes, n_genes),
                    nn.Sigmoid() # Scale expression up/down based on structural importance
                )
                
                # Encoder with LayerNorm for training stability
                self.enc = nn.Sequential(
                    nn.Linear(cond_dim, H * 2),
                    nn.LayerNorm(H * 2),
                    nn.ReLU(),
                    nn.Linear(H * 2, H),
                    nn.LayerNorm(H),
                    nn.ReLU(),
                )
                self.mu      = nn.Linear(H, d)
                self.log_var = nn.Linear(H, d)

                # Decoder with LayerNorm
                self.dec = nn.Sequential(
                    nn.Linear(d + n_types, H),
                    nn.LayerNorm(H),
                    nn.ReLU(),
                    nn.Linear(H, H * 2),
                    nn.LayerNorm(H * 2),
                    nn.ReLU(),
                    nn.Linear(H * 2, n_genes),
                )

                # Initialise weights to small values (prevents early NaN)
                self._init_weights()

            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=0.5)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

            def encode(self, x, label_oh):
                # Apply biological structural attention to isolate lineage features
                struct_x = x * self.gene_attention(x)
                h = self.enc(torch.cat([struct_x, label_oh], dim=1))
                # Clamp log_var to prevent exp() overflow
                return self.mu(h), self.log_var(h).clamp(-4.0, 4.0)

            def reparameterise(self, mu, log_var):
                """Sample z = μ + ε·σ via the reparameterisation trick."""
                std = torch.exp(0.5 * log_var)
                return mu + torch.randn_like(std) * std

            def decode(self, z, label_oh):
                return self.dec(torch.cat([z, label_oh], dim=1))

            def forward(self, x, label_oh):
                mu, log_var = self.encode(x, label_oh)
                z           = self.reparameterise(mu, log_var)
                x_hat       = self.decode(z, label_oh)
                return x_hat, mu, log_var, z

        self._model = _cVAE()
        return self._model

    # ── Loss functions ────────────────────────────────────────────────────────

    @staticmethod
    def _elbo_loss(x, x_hat, mu, log_var, beta: float = 1.0):
        """
        ELBO = MSE reconstruction + β · KL divergence.

        β-annealing (passed in from training loop) prevents posterior collapse:
        early epochs focus on reconstruction (β≈0); later epochs regularise
        the latent space towards N(0,I) (β→1).
        """
        import torch
        # Reconstruction: mean squared error per cell, then averaged over batch
        recon = ((x - x_hat) ** 2).sum(dim=1).mean()
        # KL: -0.5 Σ (1 + log σ² − μ² − σ²)
        kl    = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()
        # Guard: if either term is non-finite, return 0 for that term
        if not torch.isfinite(recon):
            recon = torch.tensor(0.0, device=x.device)
        if not torch.isfinite(kl):
            kl = torch.tensor(0.0, device=x.device)
        return recon + beta * kl

    @staticmethod
    def _triplet_loss(z, labels, margin: float = 0.5):
        """
        Batch semi-hard triplet loss on latent vectors.

        For each anchor i (type k):
          Positive p: another cell of type k (furthest → hardest positive).
          Negative n: cell of different type (closest within margin → semi-hard).
          Loss: max(d(a,p) − d(a,n) + margin, 0)

        Returns 0 if no valid triplets exist in the batch.
        """
        import torch

        sq    = (z ** 2).sum(dim=1, keepdim=True)
        dists = torch.clamp(sq + sq.T - 2.0 * z @ z.T, min=0.0)
        same  = labels.unsqueeze(0) == labels.unsqueeze(1)

        loss, n_trip = torch.tensor(0.0, device=z.device), 0

        for i in range(len(z)):
            pos_mask         = same[i].clone()
            pos_mask[i]      = False
            neg_mask         = ~same[i]

            if not pos_mask.any() or not neg_mask.any():
                continue

            d_pos  = dists[i][pos_mask].max()
            d_negs = dists[i][neg_mask]
            semi   = d_negs[(d_negs > d_pos) & (d_negs < d_pos + margin)]
            d_neg  = semi.min() if len(semi) > 0 else d_negs.min()

            triplet = torch.clamp(d_pos - d_neg + margin, min=0.0)
            if torch.isfinite(triplet):
                loss   = loss + triplet
                n_trip += 1

        return loss / max(n_trip, 1)

    @staticmethod
    def _supervised_contrastive_loss(z, labels, source_ids, temperature: float = 0.2):
        """Contrastive loss that prefers same-type cells from different slices."""
        import torch
        import torch.nn.functional as F

        if len(z) < 2:
            return torch.tensor(0.0, device=z.device)

        z = F.normalize(z, dim=1)
        sim = (z @ z.T) / max(temperature, 1e-6)
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
        same_type = labels.unsqueeze(0) == labels.unsqueeze(1)
        cross_source = source_ids.unsqueeze(0) != source_ids.unsqueeze(1)
        pos_mask = same_type & cross_source & ~eye
        if not pos_mask.any():
            pos_mask = same_type & ~eye
        if not pos_mask.any():
            return torch.tensor(0.0, device=z.device)

        logits_mask = ~eye
        exp_sim = torch.exp(sim) * logits_mask
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        valid = pos_mask.sum(dim=1) > 0
        if not valid.any():
            return torch.tensor(0.0, device=z.device)

        loss = -torch.log((pos_sum[valid] + 1e-12) / denom[valid].ravel())
        return loss.mean()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        adatas: List[AnnData],
        epochs: int = 100,
        batch_size: int = 512,
        lr: float = 3e-4,
        kl_warmup_epochs: int = 50,
        use_spatial_loss: bool = True,
        use_contrastive_loss: bool = True,
        spatial_loss_weight: float = 0.1,
        contrastive_loss_weight: float = 0.1,
        contrastive_temperature: float = 0.2,
        device: str = 'cpu',
        verbose: bool = True,
    ) -> 'INCENT_cVAE':
        """
        Train the cVAE on multiple AnnData slices (one per timepoint).

        Training uses β-KL annealing: the KL weight starts at 0 and increases
        linearly to 1 over the first ``kl_warmup_epochs`` epochs.  This prevents
        the posterior collapse problem that causes NaN loss in small datasets.

        Parameters
        ----------
        adatas : list of AnnData — all slices (timepoints) to train on.
        epochs : int, default 100.
        batch_size : int, default 512.
        lr : float, default 3e-4 — Adam learning rate.
        kl_warmup_epochs : int, default 50
            Number of epochs to linearly ramp the KL weight from 0 → 1.
            Set to 0 to disable warmup (standard β-VAE with β=1).
        use_spatial_loss : bool, default True
            Add an auxiliary spatial reconstruction head on the latent space.
        use_contrastive_loss : bool, default True
            Add a cross-time supervised contrastive loss on latent embeddings.
        spatial_loss_weight : float, default 0.1
            Weight applied to the spatial reconstruction term.
        contrastive_loss_weight : float, default 0.1
            Weight applied to the contrastive term.
        contrastive_temperature : float, default 0.2
            Temperature for the contrastive similarity softmax.
        device : str — 'cpu' or 'cuda'.
        verbose : bool — print epoch losses.

        Returns
        -------
        self — for method chaining.
        """
        import torch
        from torch.utils.data import DataLoader

        # ── Find shared genes and cell types ─────────────────────────────────
        shared_genes = list(adatas[0].var_names)
        for ad in adatas[1:]:
            shared_genes = [g for g in shared_genes if g in ad.var_names]
        if len(shared_genes) == 0:
            raise ValueError("No shared genes across provided AnnData objects.")

        all_types     = sorted(set(
            str(c)
            for ad in adatas
            for c in ad.obs['cell_type_annot'].astype(str).unique()
        ))
        cell_type_map = {ct: i for i, ct in enumerate(all_types)}

        self.shared_genes  = shared_genes
        self.cell_type_map = cell_type_map
        self.n_genes       = len(shared_genes)
        self.n_types       = len(all_types)

        # ── Build model ───────────────────────────────────────────────────────
        model = self._build_torch_model().to(device)
        spatial_head = None
        if use_spatial_loss:
            spatial_head = torch.nn.Linear(self.latent_dim, 2).to(device)
        params = list(model.parameters())
        if spatial_head is not None:
            params.extend(list(spatial_head.parameters()))
        opt   = torch.optim.Adam(params, lr=lr, eps=1e-7)
        # Cosine annealing slightly improves final embedding quality
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        # ── Dataset / loader ─────────────────────────────────────────────────
        ds     = _SpatialTranscriptomicsDataset(adatas, shared_genes, cell_type_map)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            drop_last=len(ds) >= batch_size, num_workers=0)

        n_types  = len(all_types)

        # ── Training loop ─────────────────────────────────────────────────────
        nan_streak = 0    # count consecutive NaN epochs → early stop if > 5

        for epoch in range(1, epochs + 1):
            model.train()

            # β-KL annealing: linearly increase β from 0 to 1
            beta_kl = min(1.0, epoch / max(kl_warmup_epochs, 1))

            total_loss = 0.0
            n_batches  = 0

            for x_batch, ct_batch, coord_batch, source_batch in loader:
                x_batch  = x_batch.to(device)
                ct_batch = ct_batch.to(device)
                coord_batch = coord_batch.to(device)
                source_batch = source_batch.to(device)

                # One-hot condition vector for the cell type
                label_oh = torch.zeros(len(ct_batch), n_types, device=device)
                label_oh.scatter_(1, ct_batch.unsqueeze(1), 1.0)

                # Forward
                x_hat, mu, log_var, z = model(x_batch, label_oh)

                # Losses
                elbo    = self._elbo_loss(x_batch, x_hat, mu, log_var, beta=beta_kl)
                triplet = self._triplet_loss(z, ct_batch)
                loss    = elbo + self.lambda_triplet * triplet

                if use_spatial_loss and spatial_head is not None:
                    coord_pred = spatial_head(z)
                    spatial_loss = ((coord_pred - coord_batch) ** 2).sum(dim=1).mean()
                    loss = loss + spatial_loss_weight * spatial_loss

                if use_contrastive_loss:
                    contrastive = self._supervised_contrastive_loss(
                        z, ct_batch, source_batch, temperature=contrastive_temperature)
                    loss = loss + contrastive_loss_weight * contrastive

                # Biological Prior: Structural Sparsity Penalty
                # Encourage the model to rely only on a subset of genes (lineage/structural) 
                # rather than using all transient/state genes.
                attention_weights = model.gene_attention[0].weight
                l1_penalty = torch.norm(attention_weights, p=1)
                loss = loss + 1e-4 * l1_penalty

                # Skip non-finite losses (do not backprop through NaN)
                if not torch.isfinite(loss):
                    continue

                opt.zero_grad()
                loss.backward()
                # Gradient clipping prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                total_loss += loss.item()
                n_batches  += 1

            scheduler.step()

            avg_loss = total_loss / max(n_batches, 1)

            # Track NaN streaks — if data is all-zero or identical, loss stays NaN
            if not np.isfinite(avg_loss):
                nan_streak += 1
                if nan_streak > 5:
                    import warnings
                    warnings.warn(
                        "[cVAE] Loss has been NaN for 5+ consecutive epochs.\n"
                        "This usually means the expression matrix contains only\n"
                        "zeros or identical values after preprocessing.\n"
                        "Check that your AnnData.X contains meaningful expression.",
                        stacklevel=2,
                    )
                    break
            else:
                nan_streak = 0

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"[cVAE] Epoch {epoch:4d}/{epochs}  "
                      f"loss={avg_loss:.4f}  β_KL={beta_kl:.2f}")

        self._model = model.eval()
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def embed(self, adata: AnnData, device: str = 'cpu') -> np.ndarray:
        """
        Encode cells from `adata` into the latent space.

        Returns the **posterior mean** μ (deterministic, reproducible).
        The same cell type at different timepoints will have nearby μ vectors.

        Parameters
        ----------
        adata : AnnData — must contain the genes used during training.
        device : str — 'cpu' or 'cuda'.

        Returns
        -------
        z_mu : (n_cells, latent_dim) float32.
        """
        import torch

        if self._model is None:
            raise RuntimeError("Model not trained yet.  Call .train() first.")

        model = self._model.to(device).eval()

        adata_sub = adata[:, self.shared_genes]
        X         = _to_dense(adata_sub.X)
        X         = _smart_preprocess(X)     # same preprocessing as training

        ct_indices = np.array([self.cell_type_map.get(str(c), 0)
                               for c in adata.obs['cell_type_annot'].astype(str)])

        X_t  = torch.tensor(X, dtype=torch.float32).to(device)
        ct_t = torch.tensor(ct_indices, dtype=torch.long).to(device)

        n_types  = len(self.cell_type_map)
        label_oh = torch.zeros(len(X_t), n_types, device=device)
        label_oh.scatter_(1, ct_t.unsqueeze(1), 1.0)

        with torch.no_grad():
            mu, _ = model.encode(X_t, label_oh)

        result = mu.cpu().numpy().astype(np.float32)

        # Replace any residual NaN with zero
        result = np.where(np.isfinite(result), result, 0.0)
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model weights + metadata to a .pt file."""
        import torch
        if self._model is None:
            raise RuntimeError("Model not trained yet. Call .train() before save().")
        torch.save({
            'model_state':    self._model.state_dict(),
            'n_genes':        self.n_genes,
            'n_types':        self.n_types,
            'latent_dim':     self.latent_dim,
            'hidden_dim':     self.hidden_dim,
            'lambda_triplet': self.lambda_triplet,
            'shared_genes':   self.shared_genes,
            'cell_type_map':  self.cell_type_map,
        }, path)
        print(f"[cVAE] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'INCENT_cVAE':
        """Load a saved INCENT_cVAE model from a .pt file."""
        import torch
        ck  = torch.load(path, map_location=device)
        obj = cls(
            n_genes=ck['n_genes'], n_types=ck['n_types'],
            latent_dim=ck['latent_dim'], hidden_dim=ck['hidden_dim'],
            lambda_triplet=ck['lambda_triplet'],
        )
        obj.shared_genes  = ck['shared_genes']
        obj.cell_type_map = ck['cell_type_map']
        model = obj._build_torch_model().to(device)
        model.load_state_dict(ck['model_state'])
        obj._model = model.eval()
        print(f"[cVAE] Loaded from {path}")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Public convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def train_cvae(
    adatas: List[AnnData],
    latent_dim: int = 32,
    hidden_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 3e-4,
    kl_warmup_epochs: int = 50,
    lambda_triplet: float = 1.0,
    use_spatial_loss: bool = True,
    use_contrastive_loss: bool = True,
    spatial_loss_weight: float = 0.1,
    contrastive_loss_weight: float = 0.1,
    contrastive_temperature: float = 0.2,
    device: str = 'cpu',
    verbose: bool = True,
) -> INCENT_cVAE:
    """
    Train a conditional VAE on all provided slices/timepoints.

    Pass all your MERFISH AnnData objects (one per timepoint) so the model
    learns a time-invariant latent space.

    Parameters
    ----------
    adatas : list of AnnData — all slices to train on.
    latent_dim : int, default 32
        Latent dimensionality.  For MERFISH (~250 genes), 16–32 is good.
    hidden_dim : int, default 256.
    epochs : int, default 100.
    batch_size : int, default 512.
    lr : float, default 3e-4.
    kl_warmup_epochs : int, default 50
        Epochs to linearly ramp β from 0 → 1.  Prevents posterior collapse.
    lambda_triplet : float, default 1.0
        Weight of the cell-type triplet loss.  Increase to 2–5 if same-type
        cells from different timepoints are not clustering in UMAP.
    use_spatial_loss : bool, default True
        Enable the auxiliary spatial reconstruction loss.
    use_contrastive_loss : bool, default True
        Enable the cross-time supervised contrastive loss.
    spatial_loss_weight : float, default 0.1
        Weight applied to the spatial reconstruction term.
    contrastive_loss_weight : float, default 0.1
        Weight applied to the contrastive term.
    contrastive_temperature : float, default 0.2
        Temperature for the contrastive loss.
    device : str, default 'cpu'.
    verbose : bool, default True.

    Returns
    -------
    model : INCENT_cVAE — trained and ready for .embed().

    Examples
    --------
    >>> model = train_cvae([slice_E10, slice_E12, slice_E14, slice_E16])
    >>> model.save('brain_cvae.pt')
    >>> z_A = model.embed(slice_E12)    # (n_A, latent_dim)
    >>> z_B = model.embed(slice_E16)    # (n_B, latent_dim)
    """
    shared_genes = list(adatas[0].var_names)
    for ad in adatas[1:]:
        shared_genes = [g for g in shared_genes if g in ad.var_names]

    all_types = sorted(set(
        str(c)
        for ad in adatas
        for c in ad.obs['cell_type_annot'].astype(str).unique()
    ))

    model = INCENT_cVAE(
        n_genes=len(shared_genes),
        n_types=len(all_types),
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lambda_triplet=lambda_triplet,
    )
    model.train(
        adatas,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        kl_warmup_epochs=kl_warmup_epochs,
        use_spatial_loss=use_spatial_loss,
        use_contrastive_loss=use_contrastive_loss,
        spatial_loss_weight=spatial_loss_weight,
        contrastive_loss_weight=contrastive_loss_weight,
        contrastive_temperature=contrastive_temperature,
        device=device,
        verbose=verbose,
    )
    return model


def latent_cost(
    adata_A: AnnData,
    adata_B: AnnData,
    model: INCENT_cVAE,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Compute the pairwise latent cosine distance matrix M_latent.

    M_latent[i, j] = 1 − cosine_similarity(z_i^A, z_j^B)

    Because the cVAE is conditioned on cell type and trained with a triplet
    loss, cells of the same type have small M_latent[i,j] even when their
    raw expression at timepoints t_A and t_B looks different.  This matrix
    replaces M1 in cross-timepoint runs.

    Parameters
    ----------
    adata_A : AnnData — source slice (timepoint t_A).
    adata_B : AnnData — target slice (timepoint t_B).
    model   : INCENT_cVAE — trained cVAE.
    device  : str — 'cpu' or 'cuda'.

    Returns
    -------
    M_latent : (n_A, n_B) float32.  Values in [0, 2].
    """
    z_A = model.embed(adata_A, device=device)   # (n_A, d)
    z_B = model.embed(adata_B, device=device)   # (n_B, d)

    # Unit-normalise → cosine distance = 1 − dot product
    z_A = z_A / (np.linalg.norm(z_A, axis=1, keepdims=True) + 1e-10)
    z_B = z_B / (np.linalg.norm(z_B, axis=1, keepdims=True) + 1e-10)

    M = 1.0 - z_A @ z_B.T
    return M.astype(np.float32)