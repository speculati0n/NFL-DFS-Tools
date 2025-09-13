#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
risk_model.py

Converts projections + (consistency, upside, duds) into a per-player risk model:
- sigma_base: σ in fantasy points from floor/ceiling (or position defaults)
- consistency scaling: probability within ±1σ adjusts σ (not σ itself)
- sign-dependent scales r_plus/r_minus from ±0.5σ tail rates (skew)
- sigma_eff: effective σ for covariance (RMS of sign scales)

Optimizer: use draw_optimizer_noise_points for sign-aware jitter.
Simulator: use build_covariance_from_corr + sample_skewed_outcomes for skewed sampling.
"""

from typing import Optional, Dict, Tuple, Sequence
import math
from statistics import NormalDist
import numpy as np

ND = NormalDist()
Z10 = 1.2815515655446004                 # z for 90th/10th pct
P_WITHIN_1SD = 0.6826894921370859        # P(|Z|<=1) for N(0,1)

# Safety clamps
MIN_CONSISTENCY, MAX_CONSISTENCY = 0.10, 0.95
MIN_TAIL, MAX_TAIL = 0.02, 0.48
MIN_SIGMA_POINTS, MAX_SIGMA_MULT = 0.5, 3.0

def _to_float(x) -> Optional[float]:
    try:
        if x is None: return None
        s = str(x).strip()
        if s == "": return None
        return float(s)
    except Exception:
        return None

def _prob01(x: Optional[float]) -> Optional[float]:
    p = _to_float(x)
    if p is None: return None
    if p > 1.0: p = p / 100.0
    return max(1e-6, min(1.0 - 1e-6, p))

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def estimate_sigma_base_points(
    mean_fpts: float,
    position: str,
    default_qb_var: float,
    default_skill_var: float,
    default_def_var: float,
    ceiling: Optional[float] = None,
    floor: Optional[float] = None,
) -> float:
    pos = (position or "").upper()
    m = float(mean_fpts)
    c = _to_float(ceiling)
    f = _to_float(floor)

    sigma = None
    if c is not None and f is not None and c > f:
        sigma = (c - f) / (2.0 * Z10)
    elif c is not None:
        sigma = abs(c - m) / Z10
    elif f is not None:
        sigma = abs(m - f) / Z10
    else:
        if pos == "QB":
            sigma = m * float(default_qb_var)
        elif pos == "DST":
            sigma = m * float(default_def_var)
        else:
            sigma = m * float(default_skill_var)

    sigma = max(float(MIN_SIGMA_POINTS), float(sigma))
    if sigma > MAX_SIGMA_MULT * max(1.0, m):
        sigma = MAX_SIGMA_MULT * max(1.0, m)
    return float(sigma)

def consistency_factor(consistency_raw: Optional[float], beta: float = 1.0) -> float:
    c = _prob01(consistency_raw)
    if c is None: return 1.0
    c = _clip(c, MIN_CONSISTENCY, MAX_CONSISTENCY)
    factor = (P_WITHIN_1SD / c) ** float(beta)
    return _clip(factor, 0.6, 1.6)

def _inv_phi(p: float) -> float:
    return ND.inv_cdf(p)

def calibrate_sign_scales_from_tails(
    upside_raw: Optional[float],
    duds_raw: Optional[float],
) -> Tuple[float, float]:
    up = _prob01(upside_raw)
    dn = _prob01(duds_raw)

    if up is None and dn is None:
        return 1.0, 1.0

    if up is not None:
        up = _clip(up, MIN_TAIL, MAX_TAIL)
        r_plus = 0.5 / _inv_phi(1.0 - up)
        r_plus = _clip(r_plus, 0.5, 2.0)
    else:
        r_plus = 1.0

    if dn is not None:
        dn = _clip(dn, MIN_TAIL, MAX_TAIL)
        r_minus = 0.5 / _inv_phi(1.0 - dn)
        r_minus = _clip(r_minus, 0.5, 2.0)
    else:
        r_minus = 1.0

    return float(r_plus), float(r_minus)

def sigma_effective_from_sign_scales(sigma_base: float, r_plus: float, r_minus: float) -> float:
    return float(sigma_base * math.sqrt(0.5 * (r_plus * r_plus + r_minus * r_minus)))

def build_player_risk(
    *,
    mean_fpts: float,
    position: str,
    default_qb_var: float,
    default_skill_var: float,
    default_def_var: float,
    ceiling: Optional[float] = None,
    floor: Optional[float] = None,
    consistency_raw: Optional[float] = None,
    upside_raw: Optional[float] = None,
    duds_raw: Optional[float] = None,
    beta_consistency: float = 1.0,
) -> Dict[str, float]:
    sigma_base = estimate_sigma_base_points(
        mean_fpts=mean_fpts,
        position=position,
        default_qb_var=default_qb_var,
        default_skill_var=default_skill_var,
        default_def_var=default_def_var,
        ceiling=ceiling,
        floor=floor,
    )
    c_factor = consistency_factor(consistency_raw, beta=beta_consistency)
    sigma_base_c = float(sigma_base * c_factor)
    r_plus, r_minus = calibrate_sign_scales_from_tails(upside_raw=upside_raw, duds_raw=duds_raw)
    sigma_eff = sigma_effective_from_sign_scales(sigma_base_c, r_plus, r_minus)
    return {"sigma_base": sigma_base_c, "r_plus": r_plus, "r_minus": r_minus, "sigma_eff": sigma_eff}

def draw_optimizer_noise_points(
    rng: np.random.Generator,
    randomness_pct: float,
    sigma_base: float,
    r_plus: float,
    r_minus: float,
) -> float:
    if randomness_pct <= 0:
        return 0.0
    z = rng.standard_normal()
    r_side = r_plus if z >= 0.0 else r_minus
    return float(z * sigma_base * r_side * (randomness_pct / 100.0))

def build_covariance_from_corr(
    means: Sequence[float],
    risks: Sequence[Dict[str, float]],
    corr: np.ndarray,
) -> np.ndarray:
    n = len(means)
    assert corr.shape == (n, n), "corr must be NxN"
    sig = np.array([r["sigma_eff"] for r in risks], dtype=float)
    return corr * np.outer(sig, sig)

def sample_skewed_outcomes(
    rng: np.random.Generator,
    means: Sequence[float],
    risks: Sequence[Dict[str, float]],
    corr: np.ndarray,
) -> np.ndarray:
    n = len(means)
    means = np.asarray(means, dtype=float)

    # Make correlation PSD & unit-diagonal
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(corr)
        w = np.clip(w, 0.0, None)
        corr_psd = (V @ np.diag(w) @ V.T)
        d = np.sqrt(np.clip(np.diag(corr_psd), 1e-12, None))
        corr = corr_psd / np.outer(d, d)
        L = np.linalg.cholesky(corr + 1e-10 * np.eye(n))

    z = rng.standard_normal(size=n)
    z = L @ z

    out = np.empty(n, dtype=float)
    for i, zi in enumerate(z):
        rb = risks[i]
        r_side = rb["r_plus"] if zi >= 0.0 else rb["r_minus"]
        shock = zi * rb["sigma_base"] * r_side
        out[i] = means[i] + shock

    return out
