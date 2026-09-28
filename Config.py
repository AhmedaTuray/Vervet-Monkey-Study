"""Central settings. Edit here, not in the scripts."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"      # one sub-folder per animal/session, e.g. 1139_20190219
OUTPUT_DIR = BASE_DIR / "output"  # everything generated goes here

# ---- Script 1: intensity extraction ----
# DICOM cine loops are usually log-compressed. Gamma / lognormal indicator-dilution
# models assume a LINEAR signal. Linear = 10 ** (pixel/max * DYNAMIC_RANGE_DB / 10).
LINEARIZE = False
DYNAMIC_RANGE_DB = 60.0

# If the DICOM has no frame-time tag, give the frame interval (seconds) here. Otherwise loading stops with an error.
FRAME_INTERVAL_S = None

# ---- Script 2: curve fitting ----
BASELINE_METHOD = "start"  # "start" = mean of first BASELINE_S seconds ; "min" = minimum of curve (as in your MATLAB)
BASELINE_S = 3.0
FIT_START_S = None      # e.g. 2.0 to ignore the start; None = use all
FIT_END_S = None        # e.g. 40.0 to cut off recirculation; None = use all
SMOOTH_WINDOW = 5       # frames; moving average used ONLY for initial guesses
ROBUST_LOSS = "soft_l1" # "linear" for ordinary least squares

# ---- Script 1: overlay GIF ----
GIF_STEP = 2

# ---- Script 3: groups and months (from your MATLAB script) ----
GROUPS = {
    "Control":            ["1139", "1141", "1312", "1360", "1168"],
    "Insulin Resistance": ["1251", "1375", "1216", "1293"],
    "Diabetic":           ["1436", "1242", "1431", "1367", "1102", "1118"],
}
MONTHS = {2: "February", 5: "May", 7: "July"}

# ---- Script 4: statistics ----
AG_FILE = BASE_DIR / "1_5AG.xlsx"   # columns: AnimalID, [Session YYYYMMDD], AG   (1,5-anhydroglucitol)
AG_COLUMN = "AG"
MIN_R2 = 0.70                        # model-based values from fits with R2 below this are excluded
KEY_PARAMS = ["TTP_s", "AUC", "PeakEnhancement", "WashInRate",
              "TTP_Gamma", "AUC_Gamma", "MTT_Gamma", "RBF_Gamma"]   # shown in the figures