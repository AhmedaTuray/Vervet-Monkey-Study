"""SCRIPT 4 - statistics: does CEUS follow disease state / 1,5-AG, and does modelling help?

Input : output/All_Sessions/All_Sessions_Metrics_Grouped.xlsx (script 3)
        1_5AG.xlsx (optional; columns AnimalID, [Session YYYYMMDD], AG)
Output: output/Statistics/CEUS_statistics.xlsx + figures

Design (several sessions per animal are NOT independent, so the animal is the unit):
  Q1 group differences  Kruskal-Wallis + pairwise Mann-Whitney (Holm) on animal means, Cliff's delta
  Q2 progression        ordinal trend Control < Insulin Resistance < Diabetic (Spearman on stage);
                        mixed models (random animal) for stage, group-vs-control, stage x month
  Q3 1,5-AG             animal-level Spearman + mixed model (AG, adjusted for month)
  Q4 does modelling add?  (a) fit quality of the 3 models  (b) direct vs model-derived agreement
                          (c) leave-one-animal-out classification: direct vs gamma vs other models
All p-values across parameters are also given as Benjamini-Hochberg q-values.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
from ceus import stats as S

ORDER = ["Control", "Insulin Resistance", "Diabetic"]
COLORS = {"Control": "#2b7bba", "Insulin Resistance": "#e6a100", "Diabetic": "#c0392b"}
FAMILIES = {
    "Direct": ["TTP_s", "AUC", "MTT_s", "WashInRate", "WashOutRate", "PeakEnhancement"],
    "Gamma": ["TTP_Gamma", "AUC_Gamma", "MTT_Gamma", "RBF_Gamma", "WashIn_Gamma_Model",
              "WashOut_Gamma_Model", "PeakEnhancement_Gamma", "alpha_Gamma", "beta_Gamma_s",
              "Arrival10pct_Gamma_s"],
    "Lognormal": ["TTP_Log", "AUC_Log", "MTT_Log", "RBF_Log", "WashIn_Log_Model",
                  "WashOut_Log_Model", "PeakEnhancement_Log", "Arrival10pct_Log_s"],
    "LaggedNormal": ["TTP_LagN", "AUC_LagN", "MTT_LagN", "RBF_LagN", "WashIn_LagN_Model",
                     "WashOut_LagN_Model", "PeakEnhancement_LagN", "Arrival10pct_LagN_s"],
}
MODEL_TAGS = {"_Gamma": "Gamma", "_Log": "Log", "_LagN": "LagN"}


def model_of(col):
    for tag, m in MODEL_TAGS.items():
        if tag in col:
            return m
    return None


def load_master():
    src = config.OUTPUT_DIR / "All_Sessions" / "All_Sessions_Metrics_Grouped.xlsx"
    if not src.exists():
        raise SystemExit("Run 03_group_summary.py first.")
    df = pd.read_excel(src, sheet_name="AllSessions", dtype={"Session": str, "AnimalID": str})
    df = df[df["Group"].isin(ORDER)].copy()
    if df.empty:
        raise SystemExit("No animals matched the groups in config.GROUPS.")
    df["StageNum"] = df["Group"].map({g: i for i, g in enumerate(ORDER)})
    df["MonthNum"] = pd.to_numeric(df["Month"], errors="coerce")
    return df


def apply_fit_qc(df, params):
    """Blank model-derived values whose model fit has R2 < MIN_R2 (or failed)."""
    rows = []
    for m in ("Gamma", "Log", "LagN"):
        r2c = f"R2_{m}"
        if r2c not in df:
            continue
        bad = ~(df[r2c] >= config.MIN_R2)
        for c in [p for p in params if model_of(p) == m and p in df]:
            df.loc[bad, c] = np.nan
        rows.append({"Model": m, "n_sessions": len(df), "n_excluded_R2": int(bad.sum()),
                     "median_R2": df[r2c].median(), "min_R2": df[r2c].min(),
                     "median_RMSE": df.get(f"RMSE_{m}", pd.Series(dtype=float)).median(),
                     "median_AIC": df.get(f"AIC_{m}", pd.Series(dtype=float)).median()})
    return pd.DataFrame(rows)


def compare_models(df):
    """Paired comparison of fit error (RMSE) between models on the same sessions."""
    cols = {m: f"RMSE_{m}" for m in ("Gamma", "Log", "LagN") if f"RMSE_{m}" in df}
    d = df[list(cols.values())].dropna()
    rows = []
    if len(d) >= 5 and len(cols) >= 3:
        rows.append({"Test": "Friedman (all models)", "p": stats.friedmanchisquare(*[d[c] for c in cols.values()]).pvalue,
                     "n_sessions": len(d)})
    ms = list(cols)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = d[cols[ms[i]]], d[cols[ms[j]]]
            try:
                p = stats.wilcoxon(a, b).pvalue
            except ValueError:
                p = np.nan
            rows.append({"Test": f"Wilcoxon RMSE {ms[i]} vs {ms[j]}", "p": p, "n_sessions": len(d),
                         "median_diff": float((a - b).median())})
    return pd.DataFrame(rows)


def agreement(df):
    pairs = [("TTP_s", "TTP_Gamma"), ("TTP_s", "TTP_Log"), ("TTP_s", "TTP_LagN"),
             ("AUC", "AUC_Gamma"), ("AUC", "AUC_Log"), ("AUC", "AUC_LagN"),
             ("MTT_s", "MTT_Gamma"), ("MTT_s", "MTT_Log"),
             ("PeakEnhancement", "PeakEnhancement_Gamma"), ("PeakEnhancement", "PeakEnhancement_Log"),
             ("PeakEnhancement", "PeakEnhancement_LagN"),
             ("WashInRate", "WashIn_Gamma_10_90")]
    rows = []
    for a, b in pairs:
        if a in df and b in df:
            r, p, n = S.spearman(df[a], df[b])
            rows.append({"Direct": a, "Model_derived": b, "Spearman_rho": r, "p": p, "n_sessions": n})
    return pd.DataFrame(rows)


def load_ag(df):
    if not config.AG_FILE.exists():
        print(f"  (no 1,5-AG file at {config.AG_FILE.name} - AG analyses skipped)")
        return df, False
    ag = pd.read_excel(config.AG_FILE, dtype={"AnimalID": str, "Session": str})
    ag["AnimalID"] = ag["AnimalID"].str.strip()
    ag = ag.rename(columns={config.AG_COLUMN: "AG"})
    keys = ["AnimalID", "Session"] if "Session" in ag else ["AnimalID"]
    if "Session" in ag:
        ag["Session"] = ag["Session"].str.replace(r"\.0$", "", regex=True)
    ag = ag.groupby(keys, as_index=False)["AG"].mean()
    df = df.merge(ag, on=keys, how="left")
    df["AGz"] = (df["AG"] - df["AG"].mean()) / df["AG"].std()
    print(f"  1,5-AG matched for {df['AG'].notna().sum()} of {len(df)} sessions")
    return df, df["AG"].notna().sum() >= 6


def figures(adf, out_dir, has_ag, df):
    keys = [k for k in config.KEY_PARAMS if k in adf]
    n = len(keys)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3.6 * cols, 3.4 * np.ceil(n / cols)))
    for ax, k in zip(np.atleast_1d(axes).ravel(), keys):
        data = [adf.loc[adf.Group == g, k].dropna() for g in ORDER]
        ax.boxplot(data, widths=0.5, showfliers=False)
        for i, (g, v) in enumerate(zip(ORDER, data), 1):
            ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(v)), v, color=COLORS[g], s=28, zorder=3)
        ax.set_xticks(range(1, 4), ["Ctrl", "IR", "DM"])
        ax.set_title(k, fontsize=10)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Animal means by group (Ctrl = Control, IR = Insulin Resistance, DM = Diabetic)")
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_group_boxplots.png", dpi=200)
    plt.close(fig)
    if has_ag:
        a2 = df.groupby(["AnimalID", "Group"], as_index=False)[keys + ["AG"]].mean()
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3.6 * cols, 3.4 * np.ceil(n / cols)))
        for ax, k in zip(np.atleast_1d(axes).ravel(), keys):
            for g in ORDER:
                s = a2[a2.Group == g]
                ax.scatter(s["AG"], s[k], color=COLORS[g], label=g, s=28)
            r, p, _ = S.spearman(a2["AG"], a2[k])
            ax.set_title(f"{k}\nρ={r:.2f}, p={p:.3f}", fontsize=9)
            ax.set_xlabel("1,5-AG")
        for ax in np.atleast_1d(axes).ravel()[n:]:
            ax.axis("off")
        np.atleast_1d(axes).ravel()[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "Fig_vs_15AG.png", dpi=200)
        plt.close(fig)


def main():
    out_dir = config.OUTPUT_DIR / "Statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_master()
    params = [p for fam in FAMILIES.values() for p in fam if p in df]
    print(f"{df.AnimalID.nunique()} animals, {len(df)} sessions, {len(params)} parameters")
    print(df.groupby("Group").AnimalID.nunique().reindex(ORDER).to_string())

    qc = apply_fit_qc(df, params)
    df, has_ag = load_ag(df)
    multi_month = df["MonthNum"].nunique() > 1
    adf = df.groupby(["AnimalID", "Group"], as_index=False)[params + (["AG"] if "AG" in df else [])].mean()

    # Q1 + Q2a : groups and ordinal trend on animal means
    overall, pairwise = S.group_tests(adf, params, ORDER)
    overall.insert(1, "Family", overall["Parameter"].map(
        {p: f for f, ps in FAMILIES.items() for p in ps}))

    # Q2b : mixed models, animal as random effect
    month = " + C(MonthNum)" if multi_month else ""
    specs = {"Stage (0/1/2 trend)": "StageNum" + month,
             "Group vs Control": 'C(Group, Treatment(reference="Control"))' + month}
    if multi_month:
        specs["Stage x month (progression over time)"] = "StageNum * MonthNum"
    mixed = S.mixed_models(df, params, specs)

    # Q3 : 1,5-AG
    ag_tables = {}
    if has_ag:
        rows = []
        for p in params:
            r, pv, n = S.spearman(adf["AG"], adf[p])
            rows.append({"Parameter": p, "Spearman_rho": r, "p": pv, "n_animals": n})
        ag_sp = pd.DataFrame(rows)
        ag_sp["q_BH"] = S.bh(ag_sp["p"])
        ag_tables["AG_animal_spearman"] = ag_sp
        ag_tables["AG_mixed_model"] = S.mixed_models(df, params, {"AG (z) + month": "AGz" + month})
        ag_g, _ = S.group_tests(adf, ["AG"], ORDER)
        ag_tables["AG_by_group"] = ag_g

    # Q4 : does modelling add anything?
    fitcmp = compare_models(df)
    agree = agreement(df)
    fs = {n_: [p for p in ps if p in df] for n_, ps in FAMILIES.items()}
    sets = {"Direct only": fs["Direct"], "Gamma only": fs["Gamma"],
            "Lognormal only": fs["Lognormal"], "Lagged normal only": fs["LaggedNormal"],
            "Direct + Gamma": fs["Direct"] + fs["Gamma"],
            "All models": [p for v in fs.values() for p in v]}
    clf, cms = S.loao_classification(df, sets)

    with pd.ExcelWriter(out_dir / "CEUS_statistics.xlsx") as xw:
        pd.DataFrame({"Note": [
            "Unit of analysis = animal (session values averaged) for group tests; mixed models use all sessions with a random animal intercept.",
            "Mixed-model coefficients are in SD units of the parameter (z-scored).",
            "q_BH = Benjamini-Hochberg FDR across parameters. Exploratory when n per group is small.",
            f"Model-derived values from fits with R2 < {config.MIN_R2} were excluded."]}
                     ).to_excel(xw, sheet_name="README", index=False)
        overall.to_excel(xw, sheet_name="Group_tests_trend", index=False)
        pairwise.to_excel(xw, sheet_name="Pairwise", index=False)
        mixed.to_excel(xw, sheet_name="Mixed_models", index=False)
        for k, v in ag_tables.items():
            v.to_excel(xw, sheet_name=k, index=False)
        qc.to_excel(xw, sheet_name="Fit_QC", index=False)
        fitcmp.to_excel(xw, sheet_name="Model_fit_comparison", index=False)
        agree.to_excel(xw, sheet_name="Direct_vs_model", index=False)
        clf.to_excel(xw, sheet_name="Classification", index=False)
        r0 = len(clf) + 3
        for name, cm in cms.items():
            pd.DataFrame({"": [name]}).to_excel(xw, sheet_name="Classification", startrow=r0, index=False, header=False)
            cm.to_excel(xw, sheet_name="Classification", startrow=r0 + 1)
            r0 += len(cm) + 4
        adf.to_excel(xw, sheet_name="Animal_means", index=False)
    figures(adf, out_dir, has_ag, df)

    print("\nParameters with FDR q < 0.05 (group difference or trend):")
    sig = overall[(overall.KW_q_BH < 0.05) | (overall.Trend_q_BH < 0.05)]
    print(sig[["Parameter", "KW_p", "KW_q_BH", "Trend_rho", "Trend_p", "Trend_q_BH"]].to_string(index=False)
          if len(sig) else "  none (see raw p in Group_tests_trend; treat p < 0.05 as exploratory)")
    print("\nClassification (leave-one-animal-out, chance = 0.33):")
    print(clf.to_string(index=False))
    print(f"\nSaved -> {out_dir / 'CEUS_statistics.xlsx'}")


if __name__ == "__main__":
    main()