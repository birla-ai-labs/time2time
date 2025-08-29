import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class RegimeParams:
    mu: float          # drift (per step, in log-space)
    sigma: float       # volatility (per step, in log-space)
    jump_lambda: float # expected jump count per step (0 = no jumps)
    jump_mu: float     # mean jump size (log-space)
    jump_sigma: float  # std of jump size (log-space)

CALM_DEFAULT = RegimeParams(
    mu=0.0002,         # tiny positive drift
    sigma=0.003,       # very low vol
    jump_lambda=0.0,   # no jumps
    jump_mu=0.0,
    jump_sigma=0.0
)

def crash_params(severity: float = 1.0) -> RegimeParams:
    """
    severity in [0.5, 3.0] is a good practical range.
    Higher -> more negative drift, higher vol, bigger/more-frequent negative jumps.
    """
    severity = float(severity)
    return RegimeParams(
        mu = -0.0008 * severity,        # negative drift scales with severity
        sigma = 0.008 * severity,       # volatility spike
        jump_lambda = 0.05 * severity,  # rare jumps
        jump_mu = -0.02 * severity,     # downward jump mean
        jump_sigma = 0.01 * np.sqrt(severity)  # dispersion of jump magnitudes
    )

def _simulate_regime_logpath(T: int, start_logS: float, rp: RegimeParams, rng: np.random.Generator):
    eps = rng.normal(size=T)
    # diffusion component
    diff = (rp.mu - 0.5 * (rp.sigma**2)) + rp.sigma * eps
    # jump component (compound Poisson, Gaussian jumps)
    if rp.jump_lambda > 0:
        n_jumps = rng.poisson(lam=rp.jump_lambda, size=T)
        # sum of K Gaussian jumps per step -> Gaussian with scaled variance; we sample per step
        jumps = np.where(
            n_jumps > 0,
            rng.normal(loc=rp.jump_mu * n_jumps, scale=rp.jump_sigma * np.sqrt(np.maximum(n_jumps, 1e-12))),
            0.0
        )
    else:
        jumps = np.zeros(T)
    inc = diff + jumps
    logS = np.empty(T+1)
    logS[0] = start_logS
    logS[1:] = start_logS + np.cumsum(inc)
    return logS

def generate_single_series(
    T: int = 128,
    start_price: float = 100.0,
    regime: str = "calm",     # "calm" or "crash"
    severity: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Returns tidy DataFrame with columns: ['t', 'price', 'regime', 'severity'].
    """
    rng = np.random.default_rng(seed)
    rp = CALM_DEFAULT if regime == "calm" else crash_params(severity)
    logpath = _simulate_regime_logpath(T=T, start_logS=np.log(start_price), rp=rp, rng=rng)
    price = np.exp(logpath)
    df = pd.DataFrame({
        "t": np.arange(len(price)),
        "price": price,
        "regime": regime,
        "severity": severity if regime == "crash" else 0.0
    })
    return df

def generate_regime_switch_series(
    T_calm1: int = 64,
    T_crash: int = 128,
    T_calm2: int = 64,
    start_price: float = 100.0,
    crash_severity: float = 1.5,
    seed: int = 123
) -> pd.DataFrame:
    """
    Calm -> Crash -> Calm stitched path with continuity.
    Useful for picking content/style windows for interventions.
    """
    rng = np.random.default_rng(seed)
    # first calm
    calm1 = _simulate_regime_logpath(T=T_calm1, start_logS=np.log(start_price), rp=CALM_DEFAULT, rng=rng)
    # crash starts from last calm level
    crash = _simulate_regime_logpath(T=T_crash, start_logS=calm1[-1], rp=crash_params(crash_severity), rng=rng)
    # calm2 continues from last crash level
    calm2 = _simulate_regime_logpath(T=T_calm2, start_logS=crash[-1], rp=CALM_DEFAULT, rng=rng)

    logS = np.concatenate([calm1[:-1], crash[:-1], calm2])  # avoid double-counting stitch points
    price = np.exp(logS)
    reg = (["calm"] * (T_calm1) + ["crash"] * (T_crash) + ["calm"] * (T_calm2 + 1))
    sev = ([0.0] * (T_calm1) + [crash_severity] * (T_crash) + [0.0] * (T_calm2 + 1))
    df = pd.DataFrame({
        "t": np.arange(len(price)),
        "price": price,
        "regime": reg,
        "severity": sev
    })
    return df

def make_dataset(
    n_series: int = 200,
    T: int = 128,
    p_crash: float = 0.5,
    crash_severity_range=(0.8, 2.0),
    seed: int = 7
) -> pd.DataFrame:
    """
    Creates a mixed dataset (each series is either entirely calm or crash),
    which is handy for building style pools and content pools.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_series):
        is_crash = rng.uniform() < p_crash
        if is_crash:
            sev = rng.uniform(*crash_severity_range)
            df = generate_single_series(T=T, start_price=100.0, regime="crash", severity=sev, seed=int(rng.integers(1e9)))
        else:
            df = generate_single_series(T=T, start_price=100.0, regime="calm", severity=0.0, seed=int(rng.integers(1e9)))
        df["series_id"] = sid
        rows.append(df)
    data = pd.concat(rows, ignore_index=True)
    return data
