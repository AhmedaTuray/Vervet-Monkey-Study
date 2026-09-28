"""SCRIPT 3 - master Excel with Session, AnimalID, Group and month sheets.

Folder names must contain a 4-digit animal ID and an 8-digit date (YYYYMMDD), e.g. 1139_20190219.
Groups and months are set in config.py.
Reads output/CEUS_parameters_all_cases.xlsx (script 2) ->
      output/All_Sessions/All_Sessions_Metrics_Grouped.xlsx
"""
import pandas as pd

import config


def main():
    src = config.OUTPUT_DIR / "CEUS_parameters_all_cases.xlsx"
    if not src.exists():
        raise SystemExit("Run 02_analyze_ceus.py first.")
    df = pd.read_excel(src)

    animal = df["Case"].str.extract(r"(?<!\d)(\d{4})(?!\d)")[0]
    date = df["Case"].str.extract(r"(?<!\d)(\d{8})(?!\d)")[0]
    lookup = {a: g for g, ids in config.GROUPS.items() for a in ids}
    df.insert(1, "Session", date.fillna("Unknown"))
    df.insert(2, "AnimalID", animal.fillna("Unknown"))
    df.insert(3, "Group", animal.map(lookup).fillna("Unknown"))
    df["Month"] = pd.to_datetime(date, format="%Y%m%d", errors="coerce").dt.month.astype("Int64")

    out_dir = config.OUTPUT_DIR / "All_Sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "All_Sessions_Metrics_Grouped.xlsx"
    with pd.ExcelWriter(out) as xw:
        df.to_excel(xw, sheet_name="AllSessions", index=False)
        for g in list(config.GROUPS) + ["Unknown"]:
            sub = df[df["Group"] == g]
            if len(sub):
                sub.to_excel(xw, sheet_name=g.replace(" ", ""), index=False)
                print(f"  sheet {g.replace(' ', '')}: {len(sub)} rows")
        for m, name in config.MONTHS.items():
            sub = df[df["Month"] == m]
            if len(sub):
                sub.to_excel(xw, sheet_name=name, index=False)
                print(f"  sheet {name}: {len(sub)} rows")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
