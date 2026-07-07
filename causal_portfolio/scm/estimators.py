"""OLS / 2SLS / diagnostics — Python port of the Rust cpcm-estimate crate.

Faithful port of:
  - causal_model/crates/cpcm-estimate/src/ols.rs
  - causal_model/crates/cpcm-estimate/src/tsls.rs
  - causal_model/crates/cpcm-estimate/src/diagnostics.rs

Numerics mirror the Rust exactly so results match to ~1e-9:
  - β solved via SVD pseudo-inverse with the SAME singular-value cutoff
    (threshold = 1e-10 * max singular value).
  - 2SLS first stage projects endogenous regressors onto [Z, X_exog]; second
    stage regresses y on [X̂, X_exog]; standard errors use the ORIGINAL X
    residuals (the 2SLS correction), with Var(β) = σ²(X̂'X̂)⁻¹.
  - First-stage F per endogenous var is the PARTIAL F of the excluded
    instruments: restricted (X_exog only) vs full ([Z, X_exog]) R²,
    F = ((R²f-R²r)/m) / ((1-R²f)/(n-m-k2)).
  - Sargan = n·R² of residuals-on-[Z, X_exog] (only when overidentified),
    df = m - k1 (instruments minus endogenous).
  - Hausman uses the diagonal-only simplification (sum of d²/(se_tsls²-se_ols²)
    over endogenous coefficients) with the CORRECTED 2SLS SEs; terms with a
    non-positive variance difference are skipped (difference matrix not PD),
    df = number of terms used, (None, None) if none usable.
  - HAC (Newey-West, Bartlett kernel) standard errors are computed alongside
    the iid SEs for both OLS and 2SLS; maxlags = floor(4*(n/100)^(2/9)).

Why port instead of calling statsmodels: we want bit-comparable parity with the
Rust reference (validated in tests/test_estimators.py against emitted fixtures),
and zero heavy dependencies in the hot backtest path. scipy is used only for the
Student-t / chi-squared CDFs (p-values), matching statrs in the Rust.

Validated against causal_portfolio/tests/fixtures/rust_estimator_fixtures.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2 as _chi2
from scipy.stats import t as _student_t

_SVD_REL_THRESHOLD = 1e-10  # matches the Rust `1e-10 * singular_values.max()`


# ── result containers ────────────────────────────────────────────────


@dataclass
class OlsResult:
    coefficients: np.ndarray
    std_errors: np.ndarray
    t_stats: np.ndarray
    p_values: np.ndarray
    r_squared: float
    adj_r_squared: float
    residuals: np.ndarray
    n_obs: int
    n_features: int
    feature_names: list[str]
    hac_std_errors: np.ndarray
    hac_t_stats: np.ndarray
    hac_p_values: np.ndarray


@dataclass
class TslsResult:
    coefficients: np.ndarray
    std_errors: np.ndarray
    t_stats: np.ndarray
    p_values: np.ndarray
    r_squared: float
    residuals: np.ndarray
    n_obs: int
    feature_names: list[str]
    first_stage_f: np.ndarray
    sargan_stat: float | None
    sargan_p: float | None
    hausman_stat: float | None
    hausman_p: float | None
    hac_std_errors: np.ndarray
    hac_t_stats: np.ndarray
    hac_p_values: np.ndarray


@dataclass
class Diagnostics:
    durbin_watson: float
    breusch_pagan: tuple[float, float]
    jarque_bera: tuple[float, float]
    vif: list[tuple[str, float]] = field(default_factory=list)


# ── SVD helpers (mirror ols.rs solve_via_svd / invert_via_svd) ───────


def _svd_pinv(a: np.ndarray) -> np.ndarray:
    """Pseudo-inverse via SVD with the Rust singular-value cutoff."""
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    if s.size == 0:
        return a.T.copy()
    threshold = _SVD_REL_THRESHOLD * s.max()
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    return vt.T @ np.diag(s_inv) @ u.T


def _solve_via_svd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a x = b via SVD (handles near-singular a)."""
    u, s_inv, vt = _svd_factor(a)
    return vt.T @ (s_inv * (u.T @ b))


def _svd_factor(a: np.ndarray):
    """SVD of `a` with the Rust cutoff applied, returning (u, s_inv, vt).

    Lets a caller derive BOTH the least-squares solve and the pseudo-inverse
    from a single decomposition (see `ols`, which needs both of X'X)."""
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    threshold = _SVD_REL_THRESHOLD * s.max() if s.size else 0.0
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    return u, s_inv, vt


def _t_and_p(beta: np.ndarray, std_errors: np.ndarray, df: float):
    """t-statistics and two-sided p-values (zero t where SE ~ 0)."""
    safe_se = np.where(std_errors > 1e-15, std_errors, np.inf)
    t_stats = beta / safe_se
    p_values = 2.0 * (1.0 - _student_t.cdf(np.abs(t_stats), df))
    return t_stats, np.asarray(p_values)


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Prepend a column of ones (matches ols.rs add_intercept)."""
    x = np.atleast_2d(x)
    if x.shape[0] == 1 and x.shape[1] != 1:
        x = x.T
    return np.hstack([np.ones((x.shape[0], 1)), x])


def project_onto(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """X̂ = Z (Z'Z)⁻¹ Z' X — first stage of 2SLS."""
    ztz_inv = _svd_pinv(z.T @ z)
    return z @ ztz_inv @ z.T @ x


# ── HAC (Newey-West) standard errors ─────────────────────────────────


def newey_west_maxlags(n: int) -> int:
    """Default Newey-West truncation lag: floor(4 * (n/100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _hac_covariance(
    xw: np.ndarray, residuals: np.ndarray, xtx_inv: np.ndarray, maxlags: int,
) -> np.ndarray:
    """Newey-West HAC covariance (Bartlett kernel), sandwich form.

    Var(β) = (X'X)⁻¹ S (X'X)⁻¹ with
    S = Σ_t e_t² x_t x_t' + Σ_{l=1}^{L} w_l Σ_t (x_t e_t e_{t-l} x_{t-l}' + sym),
    w_l = 1 - l/(L+1). `xw` is the bread's regressor matrix (X for OLS, the
    projected [X̂, X_exog] for 2SLS); `residuals` are the model residuals.
    No small-sample correction (matches the Rust exactly).
    """
    n = xw.shape[0]
    xu = xw * residuals.reshape(-1, 1)
    s = xu.T @ xu
    for l in range(1, min(maxlags, n - 1) + 1):
        w = 1.0 - l / (maxlags + 1.0)
        gamma = xu[l:].T @ xu[:-l]
        s = s + w * (gamma + gamma.T)
    return xtx_inv @ s @ xtx_inv


# ── OLS ──────────────────────────────────────────────────────────────


def ols(y: np.ndarray, x: np.ndarray, feature_names: list[str]) -> OlsResult:
    """Ordinary least squares with full diagnostics (port of ols.rs::ols)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, k = x.shape
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X has {x.shape[0]} rows but y has {y.shape[0]}")
    if n <= k:
        raise ValueError(f"Need n > k (got n={n}, k={k})")
    if len(feature_names) != k:
        raise ValueError(f"feature_names len ({len(feature_names)}) != X cols ({k})")

    # One SVD of X'X serves both the β solve and the (X'X)⁻¹ for SEs.
    xtx = x.T @ x
    xty = x.T @ y
    u, s_inv, vt = _svd_factor(xtx)
    beta = vt.T @ (s_inv * (u.T @ xty))
    xtx_inv = (vt.T * s_inv) @ u.T

    residuals = y - x @ beta
    sse = float(residuals @ residuals)
    y_mean = float(y.mean())
    sst = float(((y - y_mean) ** 2).sum())

    r_squared = 1.0 - sse / sst if sst > 0.0 else 0.0
    df_resid = float(n - k)
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / df_resid

    sigma2 = sse / df_resid
    std_errors = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))

    t_stats, p_values = _t_and_p(beta, std_errors, df_resid)

    hac_var = _hac_covariance(x, residuals, xtx_inv, newey_west_maxlags(n))
    hac_std_errors = np.sqrt(np.maximum(np.diag(hac_var), 0.0))
    hac_t_stats, hac_p_values = _t_and_p(beta, hac_std_errors, df_resid)

    return OlsResult(
        coefficients=beta, std_errors=std_errors, t_stats=t_stats,
        p_values=np.asarray(p_values), r_squared=r_squared,
        adj_r_squared=adj_r_squared, residuals=residuals, n_obs=n,
        n_features=k, feature_names=list(feature_names),
        hac_std_errors=hac_std_errors, hac_t_stats=hac_t_stats,
        hac_p_values=np.asarray(hac_p_values),
    )


# ── 2SLS ─────────────────────────────────────────────────────────────


def tsls(
    y: np.ndarray,
    x_endog: np.ndarray,
    x_exog: np.ndarray,
    z: np.ndarray,
    endog_names: list[str],
    exog_names: list[str],
    instrument_names: list[str],
) -> TslsResult:
    """Two-stage least squares (port of tsls.rs::tsls).

    x_exog should already include an intercept column if desired.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.shape[0]
    # Reshape to n rows directly: (n,) -> (n,1), (n,k) unchanged.
    x_endog = np.asarray(x_endog, dtype=float).reshape(n, -1)
    x_exog = np.asarray(x_exog, dtype=float).reshape(n, -1)
    z = np.asarray(z, dtype=float).reshape(n, -1)

    k1 = x_endog.shape[1]
    k2 = x_exog.shape[1]
    m = z.shape[1]
    if m < k1:
        raise ValueError(f"Need >= as many instruments ({m}) as endogenous ({k1})")

    # Stage 1: regress each endogenous var on [Z, X_exog]
    z_full = np.hstack([z, x_exog])
    x_hat = project_onto(x_endog, z_full)
    first_stage_f = _compute_first_stage_f(x_endog, z, x_exog)

    # Stage 2: regress y on [X̂, X_exog]
    x_second = np.hstack([x_hat, x_exog])
    all_names = list(endog_names) + list(exog_names)
    stage2 = ols(y, x_second, all_names)
    beta = stage2.coefficients

    # SEs use ORIGINAL X residuals (the 2SLS correction)
    x_original = np.hstack([x_endog, x_exog])
    residuals = y - x_original @ beta
    k_total = k1 + k2
    df_resid = float(n - k_total)
    sigma2 = float(residuals @ residuals) / df_resid

    xtx_hat = x_second.T @ x_second
    xtx_hat_inv = _svd_pinv(xtx_hat)
    var_beta = sigma2 * xtx_hat_inv
    std_errors = np.sqrt(np.maximum(np.diag(var_beta), 0.0))

    t_stats, p_values = _t_and_p(beta, std_errors, df_resid)

    y_mean = float(y.mean())
    sst = float(((y - y_mean) ** 2).sum())
    sse = float(residuals @ residuals)
    r_squared = 1.0 - sse / sst if sst > 0.0 else 0.0

    if m > k1:
        sargan_stat, sargan_p = _compute_sargan(y, x_original, z, x_exog, beta, k1)
    else:
        sargan_stat, sargan_p = None, None

    # Hausman compares OLS SEs with the CORRECTED 2SLS SEs (std_errors above),
    # not the raw stage-2 OLS SEs.
    ols_full = ols(y, x_original, all_names)
    hausman_stat, hausman_p = _compute_hausman(ols_full, beta, std_errors, k1)

    hac_var = _hac_covariance(x_second, residuals, xtx_hat_inv,
                              newey_west_maxlags(n))
    hac_std_errors = np.sqrt(np.maximum(np.diag(hac_var), 0.0))
    hac_t_stats, hac_p_values = _t_and_p(beta, hac_std_errors, df_resid)

    return TslsResult(
        coefficients=beta, std_errors=std_errors, t_stats=t_stats,
        p_values=np.asarray(p_values), r_squared=r_squared, residuals=residuals,
        n_obs=n, feature_names=all_names, first_stage_f=first_stage_f,
        sargan_stat=sargan_stat, sargan_p=sargan_p,
        hausman_stat=hausman_stat, hausman_p=hausman_p,
        hac_std_errors=hac_std_errors, hac_t_stats=hac_t_stats,
        hac_p_values=np.asarray(hac_p_values),
    )


def _compute_first_stage_f(
    x_endog: np.ndarray, z: np.ndarray, x_exog: np.ndarray,
) -> np.ndarray:
    """Partial F of the EXCLUDED instruments (weak-instrument diagnostic).

    Restricted: x_j on X_exog only (R²r). Full: x_j on [Z, X_exog] (R²f).
    F = ((R²f - R²r)/m) / ((1 - R²f)/(n - m - k2)).
    The omnibus F used previously let exogenous controls inflate the statistic.
    """
    n, k1 = x_endog.shape
    m = z.shape[1]
    k2 = x_exog.shape[1]
    z_full = np.hstack([z, x_exog])
    full_names = [f"c{i}" for i in range(m + k2)]
    exog_names = [f"e{i}" for i in range(k2)]
    df2 = float(n - m - k2)
    f_stats = np.zeros(k1)
    for j in range(k1):
        xj = x_endog[:, j]
        r2_r = ols(xj, x_exog, exog_names).r_squared if k2 > 0 else 0.0
        r2_f = ols(xj, z_full, full_names).r_squared
        if r2_f < 1.0 and m > 0 and df2 > 0.0:
            f_stats[j] = ((r2_f - r2_r) / m) / ((1.0 - r2_f) / df2)
        else:
            f_stats[j] = 0.0
    return f_stats


def _compute_sargan(
    y: np.ndarray, x: np.ndarray, z: np.ndarray, x_exog: np.ndarray,
    beta: np.ndarray, k1: int,
) -> tuple[float | None, float | None]:
    """Sargan overidentification: n·R² of 2SLS residuals on [Z, X_exog].

    df = m - k1 (instruments minus endogenous). The aux regression must include
    X_exog: residuals are orthogonal to exog by construction, and omitting it
    misattributes exog variation to the instruments.
    """
    residuals = y - x @ beta
    n = float(residuals.shape[0])
    m = z.shape[1]
    zx = np.hstack([z, x_exog])
    names = [f"z{i}" for i in range(m)] + [f"e{i}" for i in range(x_exog.shape[1])]
    aux = ols(residuals, zx, names)
    stat = n * aux.r_squared
    df = float(m - k1)
    if df > 0.0:
        p = float(1.0 - _chi2.cdf(stat, df))
        return float(stat), p
    return None, None


def _compute_hausman(
    ols_result: OlsResult, tsls_beta: np.ndarray, tsls_se: np.ndarray,
    k_endog: int,
) -> tuple[float | None, float | None]:
    """Diagonal Hausman using the corrected 2SLS SEs.

    Terms with var_diff <= 0 are SKIPPED (standard practice — the variance
    difference is not positive definite there); df = number of terms used.
    Returns (None, None) when no term is usable.
    """
    stat = 0.0
    df = 0
    for j in range(k_endog):
        d = float(ols_result.coefficients[j] - tsls_beta[j])
        v = float(tsls_se[j] ** 2 - ols_result.std_errors[j] ** 2)
        if v > 0.0:
            stat += d * d / v
            df += 1
    if df == 0:
        return None, None
    p = float(1.0 - _chi2.cdf(max(stat, 0.0), float(df)))
    return stat, p


# ── diagnostics (port of diagnostics.rs) ─────────────────────────────


def compute_diagnostics(
    residuals: np.ndarray, x: np.ndarray, feature_names: list[str],
) -> Diagnostics:
    return Diagnostics(
        durbin_watson=durbin_watson(residuals),
        breusch_pagan=_breusch_pagan(residuals, x, feature_names),
        jarque_bera=_jarque_bera(residuals),
        vif=_variance_inflation_factors(x, feature_names),
    )


def durbin_watson(residuals: np.ndarray) -> float:
    r = np.asarray(residuals, dtype=float).reshape(-1)
    if r.shape[0] < 2:
        return 2.0
    num = float(((r[1:] - r[:-1]) ** 2).sum())
    den = float((r ** 2).sum())
    return num / den if den > 1e-15 else 2.0


def _breusch_pagan(
    residuals: np.ndarray, x: np.ndarray, feature_names: list[str],
) -> tuple[float, float]:
    r = np.asarray(residuals, dtype=float).reshape(-1)
    n = r.shape[0]
    aux = ols(r ** 2, x, feature_names)
    stat = n * aux.r_squared
    df = float(max(x.shape[1] - 1, 1))
    p = float(1.0 - _chi2.cdf(max(stat, 0.0), df))
    return float(stat), p


def _jarque_bera(residuals: np.ndarray) -> tuple[float, float]:
    r = np.asarray(residuals, dtype=float).reshape(-1)
    n = float(r.shape[0])
    if n < 3.0:
        return 0.0, 1.0
    mean = r.mean()
    d = r - mean
    m2 = float((d ** 2).mean())
    m3 = float((d ** 3).mean())
    m4 = float((d ** 4).mean())
    if m2 < 1e-15:
        return 0.0, 1.0
    skew = m3 / m2 ** 1.5
    kurt = m4 / (m2 * m2)
    excess = kurt - 3.0
    jb = (n / 6.0) * (skew ** 2 + excess ** 2 / 4.0)
    p = float(1.0 - _chi2.cdf(max(jb, 0.0), 2.0))
    return float(jb), p


def _variance_inflation_factors(
    x: np.ndarray, feature_names: list[str],
) -> list[tuple[str, float]]:
    n, k = x.shape
    vifs: list[tuple[str, float]] = []
    for j in range(k):
        col = x[:, j]
        if np.all(np.abs(col - 1.0) < 1e-10):  # intercept
            vifs.append((feature_names[j], 1.0))
            continue
        other = [c for c in range(k) if c != j]
        if not other:
            vifs.append((feature_names[j], 1.0))
            continue
        if n <= len(other) + 1:
            vifs.append((feature_names[j], float("inf")))
            continue
        try:
            res = ols(col, x[:, other], [feature_names[c] for c in other])
            vif = 1.0 / (1.0 - res.r_squared) if res.r_squared < 1.0 else float("inf")
        except Exception:
            vif = float("inf")
        vifs.append((feature_names[j], vif))
    return vifs
