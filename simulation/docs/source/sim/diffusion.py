"""
diffusion.py — 1-D axial neutron-diffusion model of a sub-critical rod
=======================================================================

The spatial (non-point) counterpart of criticality.py.  A fissile rod of
length L is described by the steady 1-D diffusion equation with an
intrinsic (spontaneous-fission) source:

    -D ∂²φ/∂z² + Σa φ = S(z),   φ(0)=φ(L)=0  (leakage to vacuum)

whose eigenvalue equation for the fundamental mode is:
    sin(2 k L) ...   →  B² = k² + Σa/D   (material + geometric buckling)

The *critical length* L_c is where the fundamental buckling
B_c² = (π/L)² matches the material buckling.  This module gives:
  * the diffusion length L_d = sqrt(D/Σa)  (mean neutron travel before loss)
  * the critical half-length from B² = (π/(2 L_c))²
  * the steady sub-critical flux profile φ(z) for a given L

This is the teaching model behind "a bare finite assembly can be made
sub-critical simply by being too short/leaky" — the finite-size effect
that distinguishes a real piece of fissile metal from the infinite
critical-mass idealisation in criticality.py.
"""
import numpy as np

D_USUAL     = 1.5   # cm — effective diffusion coefficient (fast, teaching-scale)
SIGMA_A     = 0.10  # 1/cm — macroscopic absorption cross section (teaching-scale)


def diffusion_length():
    """L_d = sqrt(D / Σa), cm — the scale over which the flux decays."""
    return np.sqrt(D_USUAL / SIGMA_A)


def material_buckling():
    """B_m² = Σa / D — the (infinite) material buckling, 1/cm²."""
    return SIGMA_A / D_USUAL


def critical_half_length():
    """
    Critical half-length L_c where the geometric buckling matches the
    material buckling for the fundamental cosine mode:
        B_m = π / (2 L_c)   ⇒   L_c = π / (2 sqrt(B_m))
    The critical rod length is 2 * L_c.
    """
    Bm = np.sqrt(material_buckling())
    return np.pi / (2.0 * Bm)


def flux_profile(z, L):
    """
    Steady sub-critical neutron flux φ(z) in a rod of half-length L,
    vacuum boundary φ(±L)=0, intrinsic source normalised to 1:

        φ(z) = C * [ cos( (sqrt(B_m) z) ) - cos( sqrt(B_m) L ) ] / sinh(...)

    For a sub-critical rod (geometric B² < material B_m²) the flux is
    well behaved and peaks at the centre.  We use the closed-form for a
    constant source with absorption:
        φ(z) = (S0/D) * [1 - cosh(k z) / cosh(k L)]
    where k = sqrt(Σa/D) is the attenuation constant.
    """
    z = np.asarray(z, dtype=float)
    k = np.sqrt(material_buckling())
    C = np.cosh(k * L)
    phi = (1.0 - np.cosh(k * z) / C)          # shape; peak = 1 at z = 0
    return phi


def leakage_fraction(L):
    """
    Fraction of intrinsic neutrons lost by leakage from the rod surface.
    1 - (φ integrated over volume) / (infinite-medium value).  For the
    teaching model we use the ratio of the geometric to material buckling
    as a proxy:
        f_leak ≈ B_g² / B_m²  =  (π/(2L))² / B_m
    clamped to [0, 1].
    """
    Bg = np.pi / (2.0 * L)
    Bm = material_buckling()
    return float(np.clip((Bg**2) / Bm, 0.0, 1.0))


def all_outputs(L):
    """Diagnostic dump for a rod half-length L (cm)."""
    return {
        "L_cm": L,
        "L_full_cm": 2.0 * L,
        "L_d_cm": diffusion_length(),
        "B_material": material_buckling(),
        "L_crit_half_cm": critical_half_length(),
        "is_subcritical": L > critical_half_length(),
        "leakage_fraction": leakage_fraction(L),
        "center_flux_rel": 1.0,
        "edge_flux_rel": flux_profile([L], L)[0],
    }


if __name__ == "__main__":
    Ld = diffusion_length()
    Lc = critical_half_length()
    print(f"Diffusion length L_d       = {Ld:.2f} cm")
    print(f"Material buckling Bm^2     = {material_buckling():.4f} 1/cm^2")
    print(f"Critical half-length L_c   = {Lc:.2f} cm  (full rod {2*Lc:.2f} cm)")
    for L in [30.0, 50.0, 100.0]:
        r = all_outputs(L)
        print(f"  L = {L:5.1f} cm : sub-critical={r['is_subcritical']}  "
              f"leak={r['leakage_fraction']:.3f}  edge/centre phi={r['edge_flux_rel']:.3f}")
