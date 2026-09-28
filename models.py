"""Model-free metrics and shape metrics of fitted curves."""
import numpy as np
import pandas as pd

trapz = getattr(np, "trapezoid", None) or np.trapz


def moving_avg(y, w):
    return pd.Series(y).rolling(max(int(w), 1), center=True, min_periods=1).mean().to_numpy()


def r2(y, fit):
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - fit) ** 2) / ss_tot) if ss_tot > 0 else np.nan


def aic(y, fit, k):
    """Gaussian AIC from the residual sum of squares; k = number of fitted parameters."""
    n, rss = len(y), float(np.sum((y - fit) ** 2))
    return float(n * np.log(max(rss, 1e-12) / n) + 2 * k)


def rmse(y, fit):
    return float(np.sqrt(np.mean((y - fit) ** 2)))


def curve_metrics(pdf, t0, t_lo, t_hi):
    """Shape metrics of a fitted curve (without I0), evaluated on a dense grid."""
    tt = np.linspace(t0, t0 + 8 * (t_hi - t_lo), 40000)
    yy = pdf(tt)
    d = np.gradient(yy, tt)
    ipk = int(np.argmax(yy))
    Ip = float(yy[ipk])
    ry = np.maximum.accumulate(yy[:ipk + 1])
    t10 = float(np.interp(0.1 * Ip, ry, tt[:ipk + 1]))
    t90 = float(np.interp(0.9 * Ip, ry, tt[:ipk + 1]))
    below = np.where(yy[ipk:] <= 0.5 * Ip)[0]
    win = (tt >= t_lo) & (tt <= t_hi)
    return {"peak": Ip, "t_peak": float(tt[ipk]), "t10": t10,
            "wash_in_max": float(d[:ipk + 1].max()),
            "wash_in_10_90": 0.8 * Ip / max(t90 - t10, 1e-9),
            "wash_out_max": float(d[ipk:].min()),
            "half_time": float(tt[ipk + below[0]] - tt[ipk]) if len(below) else np.nan,
            "auc_window": float(trapz(yy[win], tt[win]))}


def raw_metrics(t, yb, y_abs):
    """Metrics straight from the (baseline-subtracted) data, no model."""
    ip = int(np.argmax(yb))
    peak = float(yb[ip])
    auc = float(trapz(yb, t))
    out = {"TTP_s": float(t[ip]), "AUC": auc,
           "MTT_s": float(trapz(t * yb, t) / auc) if auc else np.nan,
           "PeakIntensity": float(y_abs[ip]), "PeakEnhancement": peak}
    try:
        ry = np.maximum.accumulate(yb[:ip + 1])
        t10 = np.interp(0.1 * peak, ry, t[:ip + 1])
        t90 = np.interp(0.9 * peak, ry, t[:ip + 1])
        out["WashInRate"] = float(0.8 * peak / (t90 - t10))
    except Exception:
        out["WashInRate"] = np.nan
    out["WashOutRate"] = float(np.polyfit(t[ip:], yb[ip:], 1)[0]) if len(t) - ip >= 2 else np.nan
    out["R2_Linear"] = r2(yb, np.polyval(np.polyfit(t, yb, 1), t))
    out["R2_Smooth"] = r2(yb, moving_avg(yb, max(3, round(0.1 * len(t)))))
    return out