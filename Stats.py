"""Statistics helpers: small-n, repeated-measures aware (several sessions per animal)."""
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def bh(p):
    """Benjamini-Hochberg FDR q-values, NaN-safe."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def summary(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return ""
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return f"{med:.3g} [{q1:.3g}-{q3:.3g}]"


def group_tests(adf, params, order):
    """adf = ONE row per animal. Kruskal-Wallis, ordinal trend (Spearman on stage),
    pairwise Mann-Whitney (Holm) with Cliff's delta."""
    stage = {g: i for i, g in enumerate(order)}
    over, pair = [], []
    for prm in params:
        vals = {g: adf.loc[adf.Group == g, prm].dropna().to_numpy() for g in order}
        row = {"Parameter": prm}
        for g in order:
            row[f"n_{g}"] = len(vals[g])
            row[f"median[IQR]_{g}"] = summary(vals[g])
        try:
            h, p = stats.kruskal(*[v for v in vals.values() if len(v)])
        except ValueError:
            h, p = np.nan, np.nan
        d = adf[["Group", prm]].dropna()
        if len(d) >= 4 and d[prm].nunique() > 1:
            rho, pt = stats.spearmanr(d.Group.map(stage), d[prm])
        else:
            rho, pt = np.nan, np.nan
        row.update(KW_H=h, KW_p=p, Trend_rho=rho, Trend_p=pt)
        over.append(row)
        rows = []
        for g1, g2 in combinations(order, 2):
            a, b = vals[g1], vals[g2]
            p_mw = np.nan
            if len(a) >= 2 and len(b) >= 2:
                try:
                    p_mw = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
                except ValueError:
                    pass
            rows.append({"Parameter": prm, "Comparison": f"{g1} vs {g2}", "p_raw": p_mw,
                         "Cliffs_delta": cliffs_delta(a, b) if len(a) and len(b) else np.nan})
        ok = [i for i, r in enumerate(rows) if not np.isnan(r["p_raw"])]
        for r in rows:
            r["p_Holm"] = np.nan
        if ok:
            adj = multipletests([rows[i]["p_raw"] for i in ok], method="holm")[1]
            for i, a_ in zip(ok, adj):
                rows[i]["p_Holm"] = a_
        pair += rows
    over = pd.DataFrame(over)
    over["KW_q_BH"] = bh(over["KW_p"])
    over["Trend_q_BH"] = bh(over["Trend_p"])
    return over, pd.DataFrame(pair)


def mixed_models(df, params, specs):
    """Random intercept per animal. Parameter is z-scored, so coefficients are in SD units.
    specs = {model_name: right-hand side of the formula}"""
    rows = []
    for name, rhs in specs.items():
        for prm in params:
            d = df.copy()
            s = d[prm]
            if s.notna().sum() < 8 or s.std() == 0:
                continue
            d["y"] = (s - s.mean()) / s.std()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = smf.mixedlm("y ~ " + rhs, d, groups=d["AnimalID"]).fit(reml=True)
                ci = m.conf_int()
                for term in m.fe_params.index:
                    if term == "Intercept":
                        continue
                    rows.append({"Model": name, "Parameter": prm, "Term": term,
                                 "Coef_SD": m.fe_params[term], "SE": m.bse_fe[term],
                                 "CI_low": ci.loc[term, 0], "CI_high": ci.loc[term, 1],
                                 "p": m.pvalues[term], "n_obs": int(m.nobs),
                                 "n_animals": int(d.dropna(subset=["y"]).AnimalID.nunique())})
            except Exception:
                continue
    out = pd.DataFrame(rows)
    if len(out):
        out["q_BH"] = out.groupby(["Model", "Term"])["p"].transform(lambda x: bh(x.to_numpy()))
    return out


def spearman(x, y):
    d = pd.concat([pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)], axis=1).dropna()
    if len(d) < 4 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, len(d)
    r, p = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    return float(r), float(p), len(d)


def loao_classification(df, feature_sets, label="Group"):
    """Leave-one-ANIMAL-out multinomial logistic regression (all sessions of the held-out
    animal are unseen). Chance level for balanced accuracy = 1/number of classes."""
    rows, cms = [], {}
    animals = df["AnimalID"].unique()
    classes = sorted(df[label].unique())
    for name, feats in feature_sets.items():
        feats = [f for f in feats if f in df and df[f].notna().any()]
        if not feats:
            continue
        pred = pd.Series(index=df.index, dtype=object)
        for a in animals:
            tr, te = df[df.AnimalID != a], df[df.AnimalID == a]
            if tr[label].nunique() < 2:
                continue
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(tr[feats], tr[label])
                pred.loc[te.index] = pipe.predict(te[feats])
        ok = pred.notna()
        rows.append({"Feature_set": name, "n_features": len(feats),
                     "Balanced_accuracy": balanced_accuracy_score(df.loc[ok, label], pred[ok]),
                     "Chance": 1 / len(classes), "n_sessions": int(ok.sum())})
        cms[name] = pd.DataFrame(confusion_matrix(df.loc[ok, label], pred[ok], labels=classes),
                                 index=[f"true {c}" for c in classes],
                                 columns=[f"pred {c}" for c in classes])
    return pd.DataFrame(rows), cms