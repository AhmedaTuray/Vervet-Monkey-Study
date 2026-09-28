"""SCRIPT 1 - overlay the ROI on the DICOM and extract intensity from that region.

For every folder in data/:
  * finds the DICOM(s) by content (file extension does not matter)
  * if the scan is split (p1 / p2 ...) the parts are merged in order, p1 then p2 immediately after
  * draws the ROI on the images
  * takes the mean ROI intensity of every frame

Writes to output/<case>/
  <case>_overlay.png       first / peak / last frame with the ROI
  <case>_overlay.gif       the whole loop with the ROI outline
  <case>_intensity.xlsx    sheet TIC: Frame, Time_s, Part, Intensity (+ Intensity_linear)
                           sheet ROI_info: files used and ROI size
  <case>_TIC.png           time-intensity curve
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion

import config
from ceus.loaders import list_cases, load_case, case_out


def roi_at(d, i):
    return d["masks"][d["part"][i]]


def roi_means(d):
    f, part, masks = d["frames"], d["part"], d["masks"]
    raw = np.array([f[i][masks[part[i]]].mean() for i in range(len(f))])
    lin = None
    if config.LINEARIZE:
        # linearize each pixel first, then average (mean of logs != log of mean)
        vmax = 255.0 if f.dtype == np.uint8 else float(f.max())
        lin = np.array([(10 ** (f[i][masks[part[i]]].astype(np.float64) / vmax
                                * config.DYNAMIC_RANGE_DB / 10.0)).mean() for i in range(len(f))])
    return raw, lin


def save_overlay_png(d, raw, name, out):
    f, t = d["frames"], d["times"]
    picks = [("First", 0), ("Peak ROI intensity", int(np.argmax(raw))), ("Last", len(f) - 1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, i) in zip(axes, picks):
        m = roi_at(d, i)
        ax.imshow(f[i], cmap="gray")
        ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.35)
        ax.contour(m, levels=[0.5], colors="r", linewidths=1)
        ax.set_title(f"{title}\nframe {i}, {t[i]:.2f} s, part p{d['labels'][d['part'][i]]}")
        ax.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out / f"{name}_overlay.png", dpi=150)
    plt.close(fig)


def save_overlay_gif(d, name, out):
    f = d["frames"]                                    # original dtype: cine loops can be huge
    lo, hi = np.percentile(f[::max(1, len(f) // 40)], [1, 99.5])
    imgs = []
    for i in range(0, len(f), config.GIF_STEP):
        g = np.clip((f[i].astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)
        rgb = np.stack([g, g, g], -1)
        m = roi_at(d, i)
        rgb[m & ~binary_erosion(m)] = [1, 0, 0]
        imgs.append(Image.fromarray((rgb * 255).astype(np.uint8)))
    dt = np.median(np.diff(d["times"])) * config.GIF_STEP
    imgs[0].save(out / f"{name}_overlay.gif", save_all=True, append_images=imgs[1:],
                 duration=int(max(dt, 0.02) * 1000), loop=0)


def save_excel_and_tic(d, raw, lin, name, out):
    df = pd.DataFrame({"Frame": np.arange(len(raw)), "Time_s": d["times"],
                       "Part": [f"p{d['labels'][p]}" for p in d["part"]], "Intensity": raw})
    ycol = "Intensity"
    if lin is not None:
        df["Intensity_linear"] = lin
        ycol = "Intensity_linear"
    info = pd.DataFrame({"Part": [f"p{l}" for l in d["labels"]],
                         "DICOM_file": [a for a, _ in d["files"]],
                         "ROI_file": [b for _, b in d["files"]],
                         "ROI_pixels": [int(m.sum()) for m in d["masks"]]})
    with pd.ExcelWriter(out / f"{name}_intensity.xlsx") as xw:
        df.to_excel(xw, sheet_name="TIC", index=False)
        info.to_excel(xw, sheet_name="ROI_info", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(df.Time_s, df[ycol], "k.-", lw=0.8, ms=3)
    for p in np.where(np.diff(d["part"]) != 0)[0]:
        ax.axvline(d["times"][p + 1], color="gray", ls=":")
    ax.set(xlabel="Time (s)", ylabel=ycol.replace("_", " "), title=f"{name} - time-intensity curve")
    fig.tight_layout()
    fig.savefig(out / f"{name}_TIC.png", dpi=150)
    plt.close(fig)


def main():
    for case in list_cases():
        try:
            d = load_case(case)
            out = case_out(case.name)
            raw, lin = roi_means(d)
            save_overlay_png(d, raw, case.name, out)
            save_overlay_gif(d, case.name, out)
            save_excel_and_tic(d, raw, lin, case.name, out)
            parts = "+".join(f"p{p}" for p in d["labels"])
            print(f"[OK] {case.name}: {parts} -> {len(raw)} frames, {d['times'][-1]:.1f} s, "
                  f"image {d['frames'].shape[1:]} {d['frames'].dtype}")
            for dcm, roi in d["files"]:
                print(f"       DICOM {dcm}   ROI {roi}")
        except Exception as e:
            print(f"[SKIP] {case.name}: {e}")


if __name__ == "__main__":
    main()
