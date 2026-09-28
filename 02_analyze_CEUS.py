"""SCRIPT 2 - CEUS analysis of the time-intensity curves + Excel with the values.

Reads output/<case>/<case>_intensity.xlsx made by script 1, then
  * model-free metrics : TTP_s, AUC, MTT_s, WashInRate, WashOutRate, PeakIntensity, R2_Linear, R2_Smooth
  * gamma variate (eq. 6): AUC_Gamma, alpha, beta, t0, I0, Arrival, TTP_Gamma, MTT_Gamma, PeakIntensity_Gamma,
                           WashIn / WashOut rates, WashOut half-time, RBF_Gamma (= AUC/MTT), R2_Gamma
  * lognormal (_Log) and lagged normal (_LagN): same quantities, model-specific suffix
    (lagged-normal MTT = mean time measured from its 10 %-of-peak arrival)

Everything is ADDED to the existing files (nothing duplicated):
  <case>_intensity.xlsx : columns Gamma_fit / Lognormal_fit added to sheet TIC, new sheet Parameters
  <case>_TIC.png        : the fitted curves are drawn on the same figure
  output/CEUS_parameters_all_cases.xlsx : one row per case
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from ceus.loaders import list_cases
from ceus.metrics import moving_avg, r2, rmse, aic, curve_metrics, raw_metrics
from ceus.models import (gamma_pdf, lognormal_pdf, lagged_normal_pdf,
                         gamma_variate, lognormal, lagged_normal,
                         fit_gamma, fit_lognormal, fit_lagged_normal)


def estimate_baseline(t, y):
    if config.BASELINE_METHOD == "min":
        return float(y.min())
    return float(y[t <= t[0] + config.BASELINE_S].mean())


def analyze(case_name):
    out = config.OUTPUT_DIR / case_name
    xlsx = out / f"{case_name}_intensity.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError("run 01_overlay_and_extract.py first")
    df = pd.read_excel(xlsx, sheet_name="TIC")
    df = df[[c for c in df if c not in ("Gamma_fit", "Lognormal_fit", "LaggedNormal_fit")]]      # safe to re-run
    ycol = "Intensity_linear" if "Intensity_linear" in df else "Intensity"
    t_all, y_all = df["Time_s"].to_numpy(float), df[ycol].to_numpy(float)

    base = estimate_baseline(t_all, y_all)
    keep = np.ones(len(t_all), bool)
    if config.FIT_START_S is not None:
        keep &= t_all >= config.FIT_START_S
    if config.FIT_END_S is not None:
        keep &= t_all <= config.FIT_END_S
    t, y = t_all[keep], y_all[keep]

    ip = int(np.argmax(moving_avg(y - base, config.SMOOTH_WINDOW)))
    tp, dt = t[ip], float(np.median(np.diff(t)))
    if moving_avg(y - base, config.SMOOTH_WINDOW)[ip] <= 0 or ip < 2:
        raise ValueError("no clear peak after baseline subtraction")

    res = {"Baseline_est": base}
    res.update(raw_metrics(t, y - base, y))
    tt = np.linspace(t.min(), t.max(), 1000)

    # ---- gamma variate, eq. (6)
    pg = fit_gamma(t, y, base, tp, dt)
    auc, alpha, beta, t0, i0 = pg
    m = curve_metrics(lambda x: gamma_pdf(x, auc, alpha, beta, t0), t0, t.min(), t.max())
    mtt = beta * (alpha + 1)
    fit_g = gamma_variate(t, *pg)
    res.update({
        "AUC_Gamma": auc, "alpha_Gamma": alpha, "beta_Gamma_s": beta, "t0_Gamma_s": t0, "I0_Gamma": i0,
        "Arrival10pct_Gamma_s": m["t10"], "TTP_Gamma": t0 + alpha * beta,
        "RiseTime_Gamma_s": alpha * beta, "MTT_Gamma": mtt, "MeanTime_abs_Gamma_s": t0 + mtt,
        "PeakIntensity_Gamma": m["peak"] + i0, "PeakEnhancement_Gamma": m["peak"],
        "WashIn_Gamma_Model": m["wash_in_max"], "WashIn_Gamma_10_90": m["wash_in_10_90"],
        "WashOut_Gamma_Model": m["wash_out_max"], "WashOut_Gamma_HalfTime_s": m["half_time"],
        "AUC_Gamma_window": m["auc_window"], "RBF_Gamma": auc / mtt,
        "R2_Gamma": r2(y, fit_g), "RMSE_Gamma": rmse(y, fit_g), "AIC_Gamma": aic(y, fit_g, 5)})
    df["Gamma_fit"] = gamma_variate(t_all, *pg)
    dense = {"Gamma": gamma_variate(tt, *pg)}

    # ---- lognormal (a failure here does not stop the gamma results)
    try:
        pl = fit_lognormal(t, y, base, tp, dt)
        A, mu, sg, t0l, i0l = pl
        ml = curve_metrics(lambda x: lognormal_pdf(x, A, mu, sg, t0l), t0l, t.min(), t.max())
        mtt_l = float(np.exp(mu + sg ** 2 / 2))
        fit_l = lognormal(t, *pl)
        res.update({
            "AUC_Log": A, "mu_Log": mu, "sigma_Log": sg, "t0_Log_s": t0l, "I0_Log": i0l,
            "Arrival10pct_Log_s": ml["t10"], "TTP_Log": t0l + float(np.exp(mu - sg ** 2)),
            "MTT_Log": mtt_l, "MeanTime_abs_Log_s": t0l + mtt_l,
            "PeakIntensity_Log": ml["peak"] + i0l, "PeakEnhancement_Log": ml["peak"],
            "WashIn_Log_Model": ml["wash_in_max"], "WashIn_Log_10_90": ml["wash_in_10_90"],
            "WashOut_Log_Model": ml["wash_out_max"], "WashOut_Log_HalfTime_s": ml["half_time"],
            "AUC_Log_window": ml["auc_window"], "RBF_Log": A / mtt_l,
            "R2_Log": r2(y, fit_l), "RMSE_Log": rmse(y, fit_l), "AIC_Log": aic(y, fit_l, 5)})
        df["Lognormal_fit"] = lognormal(t_all, *pl)
        dense["Lognormal"] = lognormal(tt, *pl)
    except Exception as e:
        print(f"    lognormal fit failed: {e}")

    # ---- lagged normal (exponentially modified Gaussian)
    try:
        pn = fit_lagged_normal(t, y, base, tp, dt)
        An, mun, sgn, taun, i0n = pn
        mn = curve_metrics(lambda x: lagged_normal_pdf(x, An, mun, sgn, taun), mun - 6 * sgn,
                           t.min(), t.max())
        mean_abs = mun + taun
        mtt_n = mean_abs - mn["t10"]        # mean time measured from the 10 %-of-peak arrival
        fit_n = lagged_normal(t, *pn)
        res.update({
            "AUC_LagN": An, "mu_LagN": mun, "sigma_LagN": sgn, "tau_LagN": taun, "I0_LagN": i0n,
            "Arrival10pct_LagN_s": mn["t10"], "TTP_LagN": mn["t_peak"],
            "MeanTime_abs_LagN_s": mean_abs, "MTT_LagN": mtt_n,
            "PeakIntensity_LagN": mn["peak"] + i0n, "PeakEnhancement_LagN": mn["peak"],
            "WashIn_LagN_Model": mn["wash_in_max"], "WashIn_LagN_10_90": mn["wash_in_10_90"],
            "WashOut_LagN_Model": mn["wash_out_max"], "WashOut_LagN_HalfTime_s": mn["half_time"],
            "AUC_LagN_window": mn["auc_window"], "RBF_LagN": An / mtt_n,
            "R2_LagN": r2(y, fit_n), "RMSE_LagN": rmse(y, fit_n), "AIC_LagN": aic(y, fit_n, 5)})
        df["LaggedNormal_fit"] = lagged_normal(t_all, *pn)
        dense["LaggedNormal"] = lagged_normal(tt, *pn)
    except Exception as e:
        print(f"    lagged-normal fit failed: {e}")

    # ---- add to the existing workbook
    with pd.ExcelWriter(xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
        df.to_excel(xw, sheet_name="TIC", index=False)
        pd.DataFrame({"Parameter": list(res), "Value": list(res.values())}).to_excel(
            xw, sheet_name="Parameters", index=False)

    # ---- same figure as script 1, now with the fits
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(t_all, y_all, ".", color="lightgray", ms=4, label="data")
    ax.plot(t, y, ".", color="k", ms=4, label="data used in fit")
    ax.plot(tt, dense["Gamma"], "r-", lw=2, label=f"gamma variate (R²={res['R2_Gamma']:.3f})")
    if "Lognormal" in dense:
        ax.plot(tt, dense["Lognormal"], "b--", lw=1.5, label=f"lognormal (R²={res['R2_Log']:.3f})")
    if "LaggedNormal" in dense:
        ax.plot(tt, dense["LaggedNormal"], color="darkorange", ls="-.", lw=1.5,
                label=f"lagged normal (R²={res['R2_LagN']:.3f})")
    ax.axvline(t0, color="g", ls=":", label="t0 (gamma)")
    ax.plot(res["TTP_Gamma"], res["PeakIntensity_Gamma"], "go")
    for p in np.where(np.diff(df["Part"].astype("category").cat.codes.to_numpy()) != 0)[0]:
        ax.axvline(t_all[p + 1], color="gray", ls=":", lw=0.8)
    ax.set(xlabel="Time (s)", ylabel=ycol.replace("_", " "), title=f"{case_name} - TIC with fits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{case_name}_TIC.png", dpi=150)
    plt.close(fig)
    return res


def main():
    rows = []
    for case in list_cases():
        try:
            r = analyze(case.name)
            rows.append({"Case": case.name, **r})
            print(f"[OK] {case.name}: TTP_Gamma={r['TTP_Gamma']:.1f}s MTT_Gamma={r['MTT_Gamma']:.1f}s "
                  f"R2 gamma={r['R2_Gamma']:.3f} log={r.get('R2_Log', float('nan')):.3f} "
                  f"lagN={r.get('R2_LagN', float('nan')):.3f}")
        except Exception as e:
            print(f"[SKIP] {case.name}: {e}")
    if rows:
        out = config.OUTPUT_DIR / "CEUS_parameters_all_cases.xlsx"
        pd.DataFrame(rows).to_excel(out, index=False)
        print(f"Summary -> {out}")


if __name__ == "__main__":
    main()