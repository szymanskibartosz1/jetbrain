"""
Feature engineering — three groups of features per window:

A) per-metric stats     — mean, std, min, max, last value, slope, rate of change, skew, kurtosis
B) per-metric spectral  — dominant frequency, spectral entropy, power in low/high bands, centroid
C) cross-metric         — pairwise correlations, z-scores, cpu→latency lag

80 features total (4 metrics × 11 stat + 4 metrics × 5 spectral + 16 cross)
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from itertools import combinations
from typing import Tuple, List


METRICS = ["cpu", "memory", "latency", "error_rate"]

# Statistical features

def stat_features(w: np.ndarray) -> np.ndarray:
    """11 features from a 1-D window."""
    m = w.mean()
    s = w.std() + 1e-8

    return np.array([
        m,
        s,
        w.min(),
        w.max(),
        w[-1],
        # linear trend over the window
        np.polyfit(np.arange(len(w)), w, 1)[0],
        # how much it moved first to last
        (w[-1] - w[0]) / (len(w) - 1 + 1e-8),
        # fraction of time spent high
        skew(w),
        kurtosis(w),
        (w > np.percentile(w, 75)).mean(), # Proportion of time above 75th percentile
        w[3 * len(w) // 4:].mean() - w[: len(w) // 4].mean(), # late minus early average
    ])



STAT_FEAT_NAMES = [
    "mean", "std", "min", "max", "last",
    "slope", "roc", "skew", "kurt",
    "frac_above_p75", "late_vs_early_mean",
]


# Spectral features
def spectral_features(w: np.ndarray) -> np.ndarray:
    n = len(w)
    freqs, psd = welch(w, nperseg=min(n, 16), noverlap=min(n, 16) // 2)
    psd = psd + 1e-12

    total_power = psd.sum()

    dom_freq = freqs[np.argmax(psd)]

    psd_norm    = psd / total_power
    spec_entropy = -(psd_norm * np.log(psd_norm + 1e-12)).sum()

    mid = len(freqs) // 2
    low_power  = psd[:mid].sum() / total_power
    high_power = psd[mid:].sum() / total_power

    centroid = (freqs * psd).sum() / total_power

    return np.array([dom_freq, spec_entropy, low_power, high_power, centroid])

SPEC_FEAT_NAMES = ["dom_freq", "spec_entropy", "low_power", "high_power", "centroid"]


# Cross-metric features

def cross_features(windows: dict) -> np.ndarray:
    metric_list = list(windows.keys())
    feats = []

    # how much each pair of metrics moves togethe
    for m1, m2 in combinations(metric_list, 2):
        w1, w2 = windows[m1], windows[m2]
        s1, s2 = w1.std() + 1e-8, w2.std() + 1e-8
        corr = np.corrcoef(w1, w2)[0, 1]
        feats.append(corr if np.isfinite(corr) else 0.0)

    # how far the latest value is from the window average
    for m in metric_list:
        w = windows[m]
        z = (w[-1] - w.mean()) / (w.std() + 1e-8)
        feats.append(z)

    # biggest spike in the last 5 steps
    for m in metric_list:
        w = windows[m]
        tail = w[-5:]
        z_max = ((tail - w.mean()) / (w.std() + 1e-8)).max()
        feats.append(z_max)

    # does latency lag behind cpu? if so, by how many steps and how strongly
    cpu_w = windows["cpu"]
    lat_w = windows["latency"]\
    
    cpu_norm = (cpu_w - cpu_w.mean()) / (cpu_w.std() + 1e-8)
    lat_norm = (lat_w - lat_w.mean()) / (lat_w.std() + 1e-8)
    xcorr = np.correlate(lat_norm, cpu_norm, mode="full")
    center = len(xcorr) // 2

    lag_slice = xcorr[center - 1 : center + 6]   # lags 0..5
    feats.append(float(lag_slice.argmax()))        # lag of peak
    feats.append(float(lag_slice.max()))           # strength of peak

    return np.array(feats)



_pairs = list(combinations(METRICS, 2))
CROSS_FEAT_NAMES = (
    [f"corr_{a}_{b}" for a, b in _pairs]
    + [f"z_last_{m}" for m in METRICS]
    + [f"z_max5_{m}" for m in METRICS]
    + ["xcorr_cpu_lat_lag", "xcorr_cpu_lat_strength"])


# Master builder

def build_features(df: pd.DataFrame, metrics: List[str] = METRICS, W: int = 60, H: int = 15, step: int = 1,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Slide a window of W steps across the series and build one feature vector per position.
    Label is 1 if any incident occurs in the next H steps, 0 otherwise.
    step controls how many positions to skip between windows (1 = every step).
    """

    n = len(df)
    vals = {m: df[m].values.astype(float) for m in metrics}
    labels = df["incident_label"].values

    rows_X, rows_y, idxs = [], [], []

    for i in range(W, n - H, step):
        windows = {m: vals[m][i - W : i] for m in metrics}

        stat_block = np.concatenate([stat_features(windows[m]) for m in metrics])
        spec_block = np.concatenate([spectral_features(windows[m]) for m in metrics])
        cross_block = cross_features(windows)

        rows_X.append(np.concatenate([stat_block, spec_block, cross_block]))
        rows_y.append(int(labels[i : i + H].sum() > 0))
        idxs.append(i)

    X = np.vstack(rows_X)
    y = np.array(rows_y)
    indices = np.array(idxs)

    col_names = (
        [f"{m}_{f}" for m in metrics for f in STAT_FEAT_NAMES]
        + [f"{m}_{f}" for m in metrics for f in SPEC_FEAT_NAMES]
        + CROSS_FEAT_NAMES)


    return X, y, indices, col_names


def temporal_train_val_test_split(
    X, y, indices,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
):

    # keep time order — never shuffle
    n = len(X)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end   = int(n * (1 - test_ratio))

    return (
        X[:train_end],   y[:train_end],   indices[:train_end],
        X[train_end:val_end], y[train_end:val_end], indices[train_end:val_end],
        X[val_end:],     y[val_end:],     indices[val_end:],
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_hard_dataset import generate_hard_dataset

    df = generate_hard_dataset(n_samples=3000, n_incidents=15, seed=42)
    X, y, idx, cols = build_features(df, W=60, H=15, step=5)
    print(f"X shape: {X.shape}, y balance: {y.mean():.2%}")
    print(f"Feature({len(cols)}")
