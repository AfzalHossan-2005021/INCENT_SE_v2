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

Original INCENT functions are unchanged and still exported.
"""

# ── Original INCENT (unchanged) ───────────────────────────────────────────────
from .incent import (
    pairwise_align,
    pairwise_align_unbalanced,
    neighborhood_distribution,
    cosine_distance,
    _preprocess,
    _to_np,

    fused_gromov_wasserstein_incent,
    jensenshannon_divergence_backend,
    pairwise_msd,
    to_dense_array,
    extract_data_matrix,

# ── INCENT-SE: new alignment functions ───────────────────────────────────────
    pairwise_align_se,
    pairwise_align_spatiotemporal,
    run_incent,

# ── INCENT-SE: pose estimation ────────────────────────────────────────────────
    estimate_pose,
    apply_pose,

# ── INCENT-SE: topological fingerprints ──────────────────────────────────────
    compute_fingerprints,
    fingerprint_cost,

# ── INCENT-SE: spatial contiguity regulariser ────────────────────────────────
    build_spatial_affinity,
    augment_fgw_gradient,
    contiguity_regulariser,
    contiguity_gradient,
    estimate_overlap_fraction,

# ── INCENT-SE: cross-timepoint cVAE ──────────────────────────────────────────
    INCENT_cVAE,
    train_cvae,
    latent_cost,

# ── INCENT-SE: CAST ──────────────────────────────────────────────────────────
    pairwise_align_cast,
    compute_multiscale_descriptors,
    find_candidate_pairs,
    ransac_se2,

# ── INCENT-SE: SEOT (SE(2)-OT EM: explicit transformation recovery) ──────────
    pairwise_align_seot,
    weighted_procrustes,
    build_spatial_cost,
    seot_em,

# ── INCENT-SE: Region-Aware Partial Alignment (RAPA) ───────────────────────────────────────────────
    pairwise_align_rapa,
    decompose_target,
    match_source_to_region,
    build_anchor_cost,
    apply_rotation_only_pose,
    apply_region_translation,
    target_contiguity_gradient,

# ── INCENT-SE: BISPA (Bidirectional Iterative Spatial Pose Alignment) ───────────────────────────────────────────────
    pairwise_align_bispa,
    decompose_slice,
    build_community_similarity,
    hungarian_matching,
    recover_pose_matched,
    build_bidirectional_anchor,
    compute_overlap_fractions,

# ── INCENT-SE: LDDMM deformation ─────────────────────────────────────────────
    LDDMMDeformation,
    estimate_deformation,
    deformed_distances,
    estimate_growth_vector,

# ── GPU utilities ───────────────────────────────────────────────────────────────
    resolve_device
)

__all__ = [
    # Original INCENT
    'pairwise_align',
    'pairwise_align_unbalanced',
    'neighborhood_distribution',
    'cosine_distance',
    '_preprocess',
    '_to_np',
    'fused_gromov_wasserstein_incent',
    'jensenshannon_divergence_backend',
    'pairwise_msd',
    'to_dense_array',
    'extract_data_matrix',
    # INCENT-SE alignment
    'pairwise_align_se',
    'pairwise_align_spatiotemporal',
    'run_incent',
    'estimate_pose',
    'apply_pose',
    # Topology
    'compute_fingerprints',
    'fingerprint_cost',
    # Contiguity
    'build_spatial_affinity',
    'augment_fgw_gradient',
    'contiguity_regulariser',
    'contiguity_gradient',
    'estimate_overlap_fraction',
    # cVAE
    'INCENT_cVAE',
    'train_cvae',
    'latent_cost',
    # CAST
    'pairwise_align_cast',
    'compute_multiscale_descriptors',
    'find_candidate_pairs',
    'ransac_se2',
    # SEOT
    'pairwise_align_seot',
    'weighted_procrustes',
    'build_spatial_cost',
    'seot_em',
    # RAPA
    'pairwise_align_rapa',
    'decompose_target',
    'match_source_to_region',
    'build_anchor_cost',
    'apply_rotation_only_pose',
    'apply_region_translation',
    'target_contiguity_gradient',
    # BISPA
    'pairwise_align_bispa',
    'decompose_slice',
    'build_community_similarity',
    'hungarian_matching',
    'recover_pose_matched',
    'build_bidirectional_anchor',
    'compute_overlap_fractions',
    # LDDMM
    'LDDMMDeformation',
    'estimate_deformation',
    'deformed_distances',
    'estimate_growth_vector',
    # GPU utilities
    'resolve_device',
]