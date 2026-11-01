"""
Shared helpers for the Option-A proxy label (used by A2 and A3).

PAST-ONLY z-SCORE
-----------------
The z-score of a district-year is computed ONLY from that district's earlier years
(Year < current). Consequences:
  * every label is exactly what could have been known at that time -> rolling-origin
    validation is automatically leakage-free (no per-fold re-labelling needed);
  * the first `min_hist` years of every district have no label (the price of honesty).

Baselines
---------
  expanding : mean/variance of all earlier years
  rolling   : last `window` earlier years (follows slow drift in satellite signal)
  trend     : linear trend through earlier years, extrapolated one step (n >= 6)

Variance shrinkage
------------------
Short histories give unstable standard deviations (tiny std -> huge |z|). The variance
is shrunk toward the pooled variance of the same STATE in the same year (empirical Bayes):
    var_shr = (dof * var + k * var_pool) / (dof + k)
"""

import numpy as np
import pandas as pd


def past_stats(df, col, method="expanding", min_hist=5, window=10):
    mean = pd.Series(np.nan, index=df.index)
    var = pd.Series(np.nan, index=df.index)
    dof = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("District"):
        g = g.sort_values("Year")
        yrs = g["Year"].to_numpy(float)
        x = g[col].to_numpy(float)
        idx = g.index
        for i in range(len(g)):
            t = yrs[i]
            m = (yrs < t) & ~np.isnan(x)
            if method == "rolling":
                m &= yrs >= t - window
            xp, yp = x[m], yrs[m]
            n = len(xp)
            if n < min_hist:
                continue
            if method == "trend" and n >= max(min_hist, 6):
                b, a = np.polyfit(yp, xp, 1)
                res = xp - (a + b * yp)
                mean[idx[i]] = a + b * t
                var[idx[i]] = float((res ** 2).sum() / (n - 2))
                dof[idx[i]] = n - 2
            else:
                mean[idx[i]] = xp.mean()
                var[idx[i]] = xp.var(ddof=1)
                dof[idx[i]] = n - 1
    return mean, var, dof


def past_z(df, col, method="expanding", min_hist=5, window=10, shrink_k=4.0):
    """z = (x - past_mean) / shrunk past std.  Higher x = healthier, so damage = NEGATIVE z."""
    m, v, dof = past_stats(df, col, method, min_hist, window)
    tmp = pd.DataFrame({"State": df["State"], "Year": df["Year"], "v": v})
    pooled = tmp.groupby(["State", "Year"])["v"].transform("mean")
    v_shr = (dof * v + shrink_k * pooled) / (dof + shrink_k)
    z = (df[col] - m) / np.sqrt(v_shr)
    return z.replace([np.inf, -np.inf], np.nan)


def label_from_z(z, thr):
    """1 = damage-like decline (z <= thr), 0 = otherwise, NaN where z is unknown."""
    return (z <= thr).astype(float).where(z.notna())


def kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    ok = a.notna() & b.notna()
    if ok.sum() < 10 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
        return np.nan
    return float(cohen_kappa_score(a[ok].astype(int), b[ok].astype(int)))