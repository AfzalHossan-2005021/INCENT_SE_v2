"""
__init__.py — INCENT-SE package
================================
INCENT-SE extends the original INCENT with:

  Same-timepoint alignment (pairwise_align_se):
    - Fourier-Mellin SE(2) pose estimation
    - Topological fingerprint cost for bilateral symmetry disambiguation
    - Spatial contiguity regularisation for realistic partial overlap

  Cross-timepoint alignment (pairwise_align_spatiotemporal):
    - All of the above, plus:
    - Conditional VAE for drift-corrected expression embeddings
    - LDDMM diffeomorphic spatial deformation (BCD joint optimisation)

GPU acceleration is available in fingerprint_cost, contiguity_gradient,
all LDDMM operations, and the cVAE.  Pass use_gpu=True to any public
function and GPU is used automatically when CUDA is available.

Original INCENT functions are unchanged and still exported.
"""

# ── Original INCENT (unchanged) ───────────────────────────────────────────────
from .core import (
    pairwise_align,
    pairwise_align_unbalanced,
    neighborhood_distribution,
    cosine_distance,
    _preprocess,
    _to_np
)

from .utils import (
    fused_gromov_wasserstein_incent,
    jensenshannon_divergence_backend,
    pairwise_msd,
    to_dense_array,
    extract_data_matrix,
)

# ── INCENT-SE: new alignment functions ───────────────────────────────────────
from .core_se import (
    pairwise_align_se,
    pairwise_align_spatiotemporal,
)

from .run_incent import run_incent

# ── INCENT-SE: pose estimation ────────────────────────────────────────────────
from .pose import (
    estimate_pose,
    apply_pose,
)

# ── INCENT-SE: topological fingerprints ──────────────────────────────────────
from .topology import (
    compute_fingerprints,
    fingerprint_cost,
)

# ── INCENT-SE: spatial contiguity regulariser ────────────────────────────────
from .contiguity import (
    build_spatial_affinity,
    augment_fgw_gradient,
    contiguity_regulariser,
    contiguity_gradient,
    estimate_overlap_fraction,
)

# ── INCENT-SE: cross-timepoint cVAE ──────────────────────────────────────────
from .cvae import (
    INCENT_cVAE,
    train_cvae,
    latent_cost,
)

# ── INCENT-SE: CAST (the primary generalised alignment method) ─────────────
from .cast import (
    pairwise_align_cast,
    compute_multiscale_descriptors,
    find_candidate_pairs,
    ransac_se2,
)

from .cast_v2 import (
    pairwise_align_cast_v2,
    compute_multiscale_descriptors_v2,
    find_candidate_pairs_v2,
)

# ── INCENT-SE: SEOT (SE(2)-OT EM: explicit transformation recovery) ──────────
from .seot import (
    pairwise_align_seot,
    weighted_procrustes,
    build_spatial_cost,
    seot_em,
)

# ── INCENT-SE: RAPA (Region-Aware Partial Alignment) ─────────────────────────
from .rapa import (
    pairwise_align_rapa,
    decompose_target,
    match_source_to_region,
    build_anchor_cost,
    apply_rotation_only_pose,
    apply_region_translation,
    target_contiguity_gradient,
)

# ── INCENT-SE: BISPA (Bidirectional Symmetric Partial Alignment) ─────────────
from .bispa import (
    pairwise_align_bispa,
    decompose_slice,
    build_community_similarity,
    hungarian_matching,
    recover_pose_matched,
    build_bidirectional_anchor,
    compute_overlap_fractions,
)

# ── INCENT-SE: LDDMM deformation ─────────────────────────────────────────────
from .lddmm import (
    LDDMMDeformation,
    estimate_deformation,
    deformed_distances,
    estimate_growth_vector,
)

# ── GPU utilities (for user convenience) ─────────────────────────────────────
from ._gpu import resolve_device

__all__ = [
    # Original INCENT
    'pairwise_align', 'pairwise_align_unbalanced',
    'neighborhood_distribution', 'cosine_distance',
    'fused_gromov_wasserstein_incent', 'jensenshannon_divergence_backend',
    'pairwise_msd', 'to_dense_array', 'extract_data_matrix',
    '_preprocess', '_to_np',
    # INCENT-SE alignment
    'pairwise_align_se', 'pairwise_align_spatiotemporal',
    'run_incent',
    # Pose
    'estimate_pose', 'apply_pose',
    # Topology
    'compute_fingerprints', 'fingerprint_cost',
    # Contiguity
    'build_spatial_affinity', 'augment_fgw_gradient',
    'contiguity_regulariser', 'contiguity_gradient', 'estimate_overlap_fraction',
    # cVAE
    # CAST (the primary generalised solution)
    'pairwise_align_cast', 'compute_multiscale_descriptors',
    'find_candidate_pairs', 'ransac_se2',
    'pairwise_align_cast_v2', 'compute_multiscale_descriptors_v2',
    'find_candidate_pairs_v2',
    'INCENT_cVAE', 'train_cvae', 'latent_cost',
    # SEOT
    'pairwise_align_seot', 'weighted_procrustes', 'build_spatial_cost', 'seot_em',
    # BISPA (supersedes RAPA for the general case)
    'pairwise_align_bispa', 'decompose_slice', 'build_community_similarity',
    'hungarian_matching', 'recover_pose_matched', 'build_bidirectional_anchor',
    'compute_overlap_fractions',
    # RAPA
    'pairwise_align_rapa', 'decompose_target', 'match_source_to_region',
    'build_anchor_cost', 'apply_rotation_only_pose', 'apply_region_translation',
    'target_contiguity_gradient',
    # LDDMM
    'LDDMMDeformation', 'estimate_deformation',
    'deformed_distances', 'estimate_growth_vector',
    # GPU utility
    'resolve_device',
]