# Vervet CEUS analysis (Python / VS Code)

## Project layout
    vervet_ceus/
      config.py                    all settings
      01_overlay_and_extract.py    SCRIPT 1: ROI + DICOM overlay (p1->p2 merged) and ROI intensity -> Excel
      02_analyze_ceus.py           SCRIPT 2: CEUS parameters, gamma variate (eq. 6) + lognormal fits -> Excel
      03_group_summary.py          SCRIPT 3: Group / month sheets
      04_statistics.py             SCRIPT 4: disease-state and 1,5-AG statistics, model comparison
      1_5AG.xlsx                   (you provide) columns: AnimalID, [Session YYYYMMDD], AG
      ceus/                        helper code (loaders, models, metrics)
      data/                        your input (see below)
      output/                      everything generated

## Setup
    python -m venv .venv
    .venv\Scripts\activate          (Windows)   or   source .venv/bin/activate
    pip install -r requirements.txt
VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> pick `.venv`.

## Input layout (folder name = 4-digit animal ID + 8-digit date, any order)
    data/
      1139_20190219/
        1139_p1          <- DICOMs are found by CONTENT, any (or no) file extension works
        1139_p2             split scans: put p1 / p2 in the file name OR in a folder name (P1/, P2/)
        roi.roi          <- whole ROI (ImageJ .roi/.zip, .npy, or .png/.tif mask named *roi*/*mask*)
      1251_20190520/
        scan             <- unsplit scan
        roi.roi
If the ROI is also split, name them roi_p1.roi / roi_p2.roi.

## Run in order
    python 01_overlay_and_extract.py
    python 02_analyze_ceus.py
    python 03_group_summary.py
    python 04_statistics.py
Pipeline: DICOM+ROI -> p1/p2 alignment -> ONE intensity extraction -> ONE master TIC -> direct / gamma / lognormal / lagged-normal
parameters -> statistics (disease state + 1,5-AG).

Outputs: `output/Statistics/CEUS_statistics.xlsx` + figures; `output/<case>/`, `output/CEUS_parameters_all_cases.xlsx`, `output/All_Sessions/All_Sessions_Metrics_Grouped.xlsx`.
