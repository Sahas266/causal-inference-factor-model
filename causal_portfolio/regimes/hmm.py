"""Gaussian HMM regime classifier.

Phase A scope: fit a 2-state HMM on a small, defensible feature vector and
emit regime labels per timestep. We use a deliberately simple feature so the
regime is interpretable (risk-on / risk-off) and not circular with the
drivers we're trying to attribute returns to.

Default features:
    - VIX level (from FRED macro panel)
    - Realized vol of BTC returns (21-day rolling std, annualized)

Both are exogenous "stress" indicators. Neither is in the CPCM driver set
directly, so using them as the regime feature avoids the trap of "the regime
that best explains driver X is the regime defined by driver X."

Two label outputs:
    - `decode(features)` — viterbi over the full sequence (non-causal,
      uses future data within the sample). Fine for in-sample diagnostics.
    - `predict_causal(features)` — expanding-window one-step decode (only
      uses data up to and including each time t). Required for live use.

The classes are NOT order-invariant in state labels. After fit, we sort
states by the first feature's mean so state 0 is always the lower-stress
regime and state 1 is the higher-stress regime (assuming feature 0 increases
with stress, which VIX does). Makes the labels interpretable across runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("cpcm.regimes.hmm")


# ── feature construction ────────────────────────────────────────────


def build_regime_features(
    macro: pd.DataFrame,
    returns: pd.DataFrame,
    btc_col: str = "btc_return",
    vol_window: int = 21,
) -> pd.DataFrame:
    """Build the (VIX, BTC realized vol) feature matrix for the HMM.

    Args:
        macro: macro DataFrame with a "vixcls" column.
        returns: returns DataFrame with `btc_col` column (log returns).
        btc_col: name of the BTC return column.
        vol_window: rolling window for realized vol (default 21 days).

    Returns:
        DataFrame indexed by date with columns ["vix", "btc_vol"]. NaN rows
        dropped. Both features are z-scored against their full-sample stats
        so the HMM's Gaussian emissions are scale-invariant.
    """
    if "vixcls" not in macro.columns:
        raise KeyError("macro DataFrame must contain a 'vixcls' column")
    if btc_col not in returns.columns:
        raise KeyError(f"returns DataFrame must contain a '{btc_col}' column")

    # Realized vol: 21d rolling std of log returns, annualized
    rv = returns[btc_col].rolling(vol_window, min_periods=vol_window).std() * np.sqrt(365)
    df = pd.DataFrame({
        "vix": macro["vixcls"].reindex(returns.index).ffill(),
        "btc_vol": rv,
    }).dropna()

    # z-score so the Gaussian HMM doesn't latch onto scale
    for col in df.columns:
        mu, sigma = df[col].mean(), df[col].std()
        if sigma > 1e-9:
            df[col] = (df[col] - mu) / sigma
    return df


# ── classifier ──────────────────────────────────────────────────────


@dataclass
class _FittedHMM:
    """Snapshot of the fitted HMM with label-permutation already applied."""
    model: object  # GaussianHMM
    permutation: np.ndarray  # state_old → state_new


class RegimeClassifier:
    """Wraps GaussianHMM with deterministic state labeling and causal decode."""

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 200,
        random_state: int = 42,
        covariance_type: str = "full",
        n_restarts: int = 1,
    ):
        """Args:
            n_states: number of regimes (2 or 3 typical).
            n_iter: max EM iterations per fit.
            random_state: base seed; restarts use random_state, random_state+1, ...
            covariance_type: 'full', 'diag', or 'tied'.
            n_restarts: number of seeds to try in fit(); the highest-likelihood
                result wins. Use ≥5 for n_states ≥ 3 (we've observed 60% of
                seeds finding bad local optima for 3-state on this data).
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.n_restarts = n_restarts
        self._fitted: _FittedHMM | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted is not None

    def fit(self, features: pd.DataFrame | np.ndarray) -> "RegimeClassifier":
        """Fit (with multi-start if n_restarts > 1) and apply state relabeling.

        Multi-start strategy: try `n_restarts` seeds and keep the model with
        the highest log-likelihood. Fixes a real issue we observed empirically
        — for 3-state HMM on (VIX, BTC_vol), the default seed found a bad
        local optimum (degenerate states flapping daily) while other seeds
        found a stable, persistent 3-regime structure with much higher
        likelihood.
        """
        from hmmlearn.hmm import GaussianHMM

        X = np.asarray(features)
        if X.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {X.shape}")

        best_model = None
        best_ll = -np.inf
        for restart in range(self.n_restarts):
            seed = self.random_state + restart
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=seed,
                )
                model.fit(X)
                ll = model.score(X)
                if ll > best_ll:
                    best_ll = ll
                    best_model = model
            except Exception as e:
                logger.warning("HMM fit failed for seed=%d: %s", seed, e)

        if best_model is None:
            raise RuntimeError(
                f"All {self.n_restarts} HMM fit restarts failed"
            )

        # Sort states by feature-0 mean so state 0 is always the lower-stress regime.
        order = np.argsort(best_model.means_[:, 0])
        permutation = np.empty_like(order)
        permutation[order] = np.arange(self.n_states)

        self._fitted = _FittedHMM(model=best_model, permutation=permutation)
        logger.info(
            "HMM fitted: %d states, ll=%.1f (%d restarts), means (feat0)=%s",
            self.n_states, best_ll, self.n_restarts,
            [round(m, 3) for m in best_model.means_[order, 0].tolist()],
        )
        return self

    def _relabel(self, raw_labels: np.ndarray) -> np.ndarray:
        assert self._fitted is not None
        return self._fitted.permutation[raw_labels]

    def decode(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Viterbi decode over the full sequence (non-causal, uses future)."""
        if self._fitted is None:
            raise RuntimeError("fit() before decode()")
        X = np.asarray(features)
        _, raw = self._fitted.model.decode(X)
        return self._relabel(raw)

    def predict_causal(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """One-step-at-a-time decode using only data up to and including each t.

        Slower than `decode` (O(T²) vs O(T)) but does not leak future info.
        """
        if self._fitted is None:
            raise RuntimeError("fit() before predict_causal()")
        X = np.asarray(features)
        T = len(X)
        labels = np.empty(T, dtype=int)
        for t in range(T):
            _, raw = self._fitted.model.decode(X[: t + 1])
            labels[t] = self._fitted.permutation[raw[-1]]
        return labels

    def forward_filter(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Causal posterior P(state_t | observations[0:t+1]) for each t.

        This is the forward algorithm output (filtered, not smoothed). It is
        strictly causal: posterior[t] uses only data[0:t+1]. Contrast with
        the standard HMM `predict_proba` which returns the *smoothed*
        posterior P(state_t | full sequence) — non-causal.

        Returns:
            (T, n_states) array of state probabilities.
        """
        if self._fitted is None:
            raise RuntimeError("fit() before forward_filter()")
        X = np.asarray(features)
        model = self._fitted.model

        # _compute_log_likelihood returns (T, n_states) log emission probs
        log_emit = model._compute_log_likelihood(X)
        log_trans = np.log(np.maximum(model.transmat_, 1e-300))
        log_start = np.log(np.maximum(model.startprob_, 1e-300))

        T, K = log_emit.shape
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = log_start + log_emit[0]
        for t in range(1, T):
            # log_alpha[t, k] = log(sum_j exp(log_alpha[t-1, j] + log_trans[j, k])) + log_emit[t, k]
            # Use logsumexp trick
            a = log_alpha[t - 1][:, None] + log_trans  # (K, K)
            log_alpha[t] = _logsumexp(a, axis=0) + log_emit[t]

        # Normalize each row to get posterior; then apply state permutation
        log_norm = _logsumexp(log_alpha, axis=1, keepdims=True)
        posterior_raw = np.exp(log_alpha - log_norm)

        # Permute columns so state 0 is always lower-stress
        p_inv = np.argsort(self._fitted.permutation)
        return posterior_raw[:, p_inv]

    def predict_forward(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Hard labels from forward filter (argmax of causal posterior)."""
        return self.forward_filter(features).argmax(axis=1)

    def transition_matrix(self) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("fit() before transition_matrix()")
        # Permute rows AND columns so the matrix uses the sorted labels.
        m = self._fitted.model.transmat_
        p = self._fitted.permutation
        # P_new[i, j] = P_old[p_inv[i], p_inv[j]]
        p_inv = np.argsort(p)
        return m[p_inv][:, p_inv]

    def state_means(self) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("fit() before state_means()")
        m = self._fitted.model.means_
        return m[np.argsort(self._fitted.permutation)]


# ── numerical helper ─────────────────────────────────────────────────


def _logsumexp(a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log(sum(exp(a))). Avoids underflow at very negative values."""
    a_max = np.max(a, axis=axis, keepdims=True)
    # Replace -inf maxes with 0 so we don't get nan
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    return out + a_max


# ── rolling-window fit+decode (full causal pipeline) ─────────────────


def rolling_fit_decode(
    features: pd.DataFrame,
    window_size: int,
    refit_every: int = 21,
    n_states: int = 2,
    n_restarts: int = 5,
    random_state: int = 42,
    covariance_type: str = "full",
) -> pd.Series:
    """Fully causal regime labels via rolling-window fit + forward filter.

    For each timestep t >= window_size:
      1. If t is a refit boundary (t % refit_every == 0 since first valid t),
         refit the HMM on features[t - window_size : t].
      2. Forward-filter the window and take the label at the last index.
      3. (Optional) re-use the previously-fit model between refit boundaries
         for speed — its parameters are still based only on past data.

    Args:
        features: (T, D) DataFrame of regime features, datetime-indexed.
        window_size: how many trailing days to fit on (e.g. 252).
        refit_every: refit cadence in days. Smaller = more adaptive, slower.
        n_states, n_restarts, random_state, covariance_type: passed to
            `RegimeClassifier`.

    Returns:
        pd.Series of integer labels, indexed by features.index. The first
        `window_size` entries are NaN (not enough history).
    """
    T = len(features)
    labels = np.full(T, np.nan)
    if T < window_size:
        return pd.Series(labels, index=features.index, name="regime")

    classifier: RegimeClassifier | None = None
    last_refit = -np.inf

    for t in range(window_size, T):
        if classifier is None or (t - last_refit) >= refit_every:
            window = features.iloc[t - window_size : t]
            classifier = RegimeClassifier(
                n_states=n_states, random_state=random_state,
                covariance_type=covariance_type, n_restarts=n_restarts,
            ).fit(window)
            last_refit = t

        # Forward-filter the most recent `window_size` observations
        window = features.iloc[t - window_size + 1 : t + 1]
        labels[t] = classifier.predict_forward(window)[-1]

    return pd.Series(labels, index=features.index, name="regime").astype("Int64")


# ── dwell-time summaries ─────────────────────────────────────────────


def dwell_stats(labels: np.ndarray) -> dict[int, dict]:
    """Per-state: total days, % of sample, n_runs, mean run length."""
    out: dict[int, dict] = {}
    if len(labels) == 0:
        return out

    # Find run boundaries
    runs: list[tuple[int, int]] = []  # (state, length)
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            runs.append((int(labels[start]), i - start))
            start = i
    runs.append((int(labels[start]), len(labels) - start))

    for state in sorted(set(int(s) for s in labels)):
        state_runs = [length for s, length in runs if s == state]
        out[state] = {
            "days": int(sum(state_runs)),
            "pct": float(sum(state_runs)) / len(labels),
            "n_runs": len(state_runs),
            "mean_run_length": float(np.mean(state_runs)) if state_runs else 0.0,
            "max_run_length": int(max(state_runs)) if state_runs else 0,
        }
    return out
