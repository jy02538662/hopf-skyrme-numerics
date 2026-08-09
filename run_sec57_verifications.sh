# Section 5.7 numerical verifications
#
# These three scripts stress the C2(z) (far-field antisymmetry) representation
# argument for the Q=2 double-ring orthogonal configuration.  Run them on a
# GPU box (the trilinear interpolation is identical in style to
# farfield_multipole.py so the CPU fallback is fine but slower).
#
# Recommended order:
#   (a) d2_symmetry_probe.py   -> confirms Assumption 5.A on Q=2
#                                  (delta_n_perp -> -delta_n_perp under C2(z))
#                                  Same script on Q=1 as control.
#   (b) sh_decompose_components.py -> confirms Lemma 5.4 (signed l=0 ~ 0)
#                                  and addresses gap 2 (chi monopole behavior)
#                                  Same on Q=1 as control (l=0 NOT forced 0).
#   (c) q2_symmetry_full_audit.py (vacuum-preserving variant)
#                                  -> pins down the *bulk* symmetry group:
#                                     D2 vs D2h vs D2d vs C2.
#                                  Result (after v2 fix): C2 only -- see
#                                  notes/roadmap_v2/sec57_strict_rewrite_v1.md
#                                  sec 5.7.2 / 5.7.9.

# ---- Q=2 representative final field (N=96, L=12, p=1,q=2,scale=3.0) ----
# From README.md Phase 2 logs:
#   outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy
# E ≈ 560.940, Q_fft ≈ -1.875251, N=96, L=12.

Q2_FIELD=outputs/Q2_p1q2_N96_L12_s30_qpen0_short/nfield_q1_torch.npy
Q2_LEN=12

# ---- Q=1 control baseline (N=80, L=10, unconstrained long run) ----
Q1_FIELD=outputs/q1_torch_N80_L10_s18_qpen0_long/nfield_q1_torch.npy
Q1_LEN=10

mkdir -p outputs/sec57_d2_probe_Q2 outputs/sec57_shdecomp_Q2 outputs/sec57_d2_probe_Q1

# (a) D2 antisymmetry probe on Q=2
python src/d2_symmetry_probe.py \
    --field "$Q2_FIELD" \
    --length $Q2_LEN \
    --q-shells 3 4 5 6 7 8 9 10 \
    --n-points 6000 \
    --out outputs/sec57_d2_probe_Q2

# (a') Same on Q=1 as a control (Q=1 has axial symmetry + reflection,
#      NOT D2; should fail / be partially symmetric)
python src/d2_symmetry_probe.py \
    --field "$Q1_FIELD" \
    --length $Q1_LEN \
    --q-shells 2 3 4 5 6 7 8 \
    --n-points 6000 \
    --out outputs/sec57_d2_probe_Q1

# (b) Per-component SH decomposition on Q=2 (lmax=4 to also probe l=4 block
#     for Q=3/Q=4 audits when those become available)
python src/sh_decompose_components.py \
    --field "$Q2_FIELD" \
    --length $Q2_LEN \
    --r-shells 3 4 5 6 7 8 9 10 \
    --n-points 8000 \
    --lmax 4 \
    --out outputs/sec57_shdecomp_Q2

# (b') Same on Q=1 for control.  Expected: dnx_l0, dny_l0 are NOT forced to
#      zero (Q=1 axial symmetry allows l=0); even-m block IS partially present
#      but not as null as Q=2.
python src/sh_decompose_components.py \
    --field "$Q1_FIELD" \
    --length $Q1_LEN \
    --r-shells 2 3 4 5 6 7 8 \
    --n-points 8000 \
    --lmax 4 \
    --out outputs/sec57_shdecomp_Q1

# (c.0) Full symmetry group audit for Q=2.  This pins down whether the Q=2
#       compact configuration is in D2 (Klein four-group, no mirrors), D2h
#       (added mirrors), D2d (added S4), or only the smaller C2 (just one
#       180-degree rotation).  The expected verdict, given the published
#       double-ring orthogonal picture and the far-field C2(z) antisymmetry
#       we already established, is: D2 OR C2 -- not D2h, not D2d.
#       Result (v2, vacuum-preserving variant): the field is in C2 (only
#       C2(z) survives as a bulk symmetry).  See notes/roadmap_v2/
#       sec57_strict_rewrite_v1.md, sec 5.7.2 and 5.7.9.
python src/q2_symmetry_full_audit.py \
    --field "$Q2_FIELD" \
    --length $Q2_LEN \
    --tolerance 1e-2 \
    --out outputs/sec57_symm_audit_Q2

# (c) (Optional) Per-component SH on Q=3 and Q=4 if those field files exist.
#     These probes test the Qeff=0 / alpha=5 or 7 prediction.
if [ -f outputs/Q3_L12_phase2/nfield_q3_torch.npy ]; then
    mkdir -p outputs/sec57_shdecomp_Q3
    python src/sh_decompose_components.py \
        --field outputs/Q3_L12_phase2/nfield_q3_torch.npy \
        --length 12 \
        --r-shells 3 4 5 6 7 8 9 10 \
        --n-points 8000 \
        --lmax 4 \
        --out outputs/sec57_shdecomp_Q3
fi

if [ -f outputs/Q4_L12_phase2/nfield_q4_torch.npy ]; then
    mkdir -p outputs/sec57_shdecomp_Q4
    python src/sh_decompose_components.py \
        --field outputs/Q4_L12_phase2/nfield_q4_torch.npy \
        --length 12 \
        --r-shells 3 4 5 6 7 8 9 10 \
        --n-points 8000 \
        --lmax 4 \
        --out outputs/sec57_shdecomp_Q4
fi