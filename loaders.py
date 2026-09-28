"""Find and read DICOMs (any file extension) and ROIs; merge p1 -> p2 -> ..."""
import re
import warnings
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from skimage.draw import polygon

import config

# extensions that are never a DICOM scan (everything else is checked by content)
NON_DICOM_EXT = {".roi", ".zip", ".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
                 ".bmp", ".gif", ".xlsx", ".xls", ".csv", ".txt", ".py", ".json", ".pdf",
                 ".docx", ".md", ".mat", ".mp4", ".avi", ".mov"}
PART_RE = re.compile(r"(?:^|[_\-\s.])p(\d+)(?:$|[_\-\s.])")


# ------------------------------------------------------------------ file discovery
def is_dicom(path):
    """True if the file *content* is DICOM - the file name / extension does not matter."""
    p = Path(path)
    if p.name.startswith(".") or p.suffix.lower() in NON_DICOM_EXT:
        return False
    try:
        with open(p, "rb") as f:
            if f.read(132)[128:132] == b"DICM":
                return True
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)   # files without the preamble
        return "Rows" in ds and "Columns" in ds
    except Exception:
        return False


def is_roi(path):
    p = Path(path)
    e = p.suffix.lower()
    if p.name.startswith("."):
        return False
    if e in {".roi", ".zip", ".npy"}:
        return True
    # image masks must say roi/mask in the name so screenshots are not mistaken for ROIs
    return e in {".png", ".tif", ".tiff", ".bmp"} and any(k in p.stem.lower() for k in ("roi", "mask"))


def part_number(path, root=None):
    """p-number from the file name, else from its folders below `root` ('P1/IMG001' -> 1)."""
    path = Path(path)
    names = [path.name]
    if root is not None:
        try:
            names += list(path.relative_to(root).parts[:-1])[::-1]
        except ValueError:
            pass
    for n in names:
        m = PART_RE.search(n.lower())
        if m:
            return int(m.group(1))
    return None


def list_cases():
    if not config.DATA_DIR.exists():
        raise SystemExit(f"Data folder not found: {config.DATA_DIR}")
    return sorted(p for p in config.DATA_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))


def find_inputs(case_dir):
    case_dir = Path(case_dir)
    files = [p for p in case_dir.rglob("*") if p.is_file()
             and not any(x.startswith(".") for x in p.relative_to(case_dir).parts)]
    dicoms = [p for p in files if is_dicom(p)]
    rois = [p for p in files if is_roi(p)]
    if not dicoms:
        raise FileNotFoundError(f"no DICOM files found in {case_dir.name}")
    if not rois:
        raise FileNotFoundError(f"no ROI file found in {case_dir.name}")
    nums = [part_number(d, case_dir) for d in dicoms]
    if len(dicoms) > 1 and any(n is None for n in nums):
        raise ValueError(f"{case_dir.name}: {len(dicoms)} DICOMs found but not all are marked p1/p2 "
                         f"(in file or folder name): {[d.name for d in dicoms]}")
    dicoms = [d for _, d in sorted(zip([n or 0 for n in nums], dicoms), key=lambda x: (x[0], str(x[1])))]
    nums = [part_number(d, case_dir) for d in dicoms]
    if len(dicoms) > 1 and nums != list(range(1, len(dicoms) + 1)):
        warnings.warn(f"{case_dir.name}: parts found are {nums}; expected 1..{len(dicoms)}")
    if len(dicoms) == 1 and nums[0] is not None:
        warnings.warn(f"{case_dir.name}: only p{nums[0]} found - is the other part missing?")
    return dicoms, rois


def pick_roi(rois, part, case_dir):
    """Prefer an ROI with the same p-number; otherwise the shared (whole) ROI."""
    if part is not None:
        specific = [r for r in rois if part_number(r, case_dir) == part]
        if specific:
            return specific[0]
    common = [r for r in rois if part_number(r, case_dir) is None]
    if common:
        return common[0]
    if len(rois) == 1:
        return rois[0]
    raise ValueError(f"cannot decide which ROI belongs to part {part}: {[r.name for r in rois]}")


# ------------------------------------------------------------------ reading
def load_dicom(path):
    """Return frames (N,H,W) and frame times in seconds starting at 0."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    if int(getattr(ds, "SamplesPerPixel", 1)) == 3:          # colour -> grey
        arr = (arr[..., :3] @ np.array([0.299, 0.587, 0.114])).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    n = arr.shape[0]

    if getattr(ds, "FrameTimeVector", None) is not None:
        t = np.cumsum(np.array(ds.FrameTimeVector, dtype=float)) / 1000.0
        t = t - t[0]
    else:
        if getattr(ds, "FrameTime", None):
            dt = float(ds.FrameTime) / 1000.0
        elif getattr(ds, "CineRate", None):
            dt = 1.0 / float(ds.CineRate)
        elif getattr(ds, "RecommendedDisplayFrameRate", None):
            dt = 1.0 / float(ds.RecommendedDisplayFrameRate)
        elif config.FRAME_INTERVAL_S:
            dt = float(config.FRAME_INTERVAL_S)
        else:
            raise ValueError(f"{Path(path).name}: no frame-time tag in the DICOM. "
                             "Set FRAME_INTERVAL_S in config.py")
        t = np.arange(n) * dt
    return arr, t


def load_roi_mask(path, shape):
    ext = Path(path).suffix.lower()
    if ext in {".roi", ".zip"}:
        import roifile
        rois = roifile.roiread(str(path))
        rois = rois if isinstance(rois, list) else [rois]
        mask = np.zeros(shape, bool)
        for r in rois:
            xy = r.coordinates()
            rr, cc = polygon(xy[:, 1], xy[:, 0], shape)
            mask[rr, cc] = True
    elif ext == ".npy":
        mask = np.load(path) > 0
    else:
        mask = np.array(Image.open(path).convert("L")) > 0
    if mask.ndim == 3:
        mask = mask.any(axis=0)
    if mask.shape != tuple(shape):
        raise ValueError(f"ROI {Path(path).name} is {mask.shape}, image is {tuple(shape)}")
    if not mask.any():
        raise ValueError(f"ROI {Path(path).name} is empty")
    return mask


def load_case(case_dir):
    """Load every part, attach its ROI, merge p1 -> p2 -> ... into one continuous series.

    A later part starts one frame interval after the previous part's last frame,
    so the merged time axis has no gap and no shared time point.
    """
    case_dir = Path(case_dir)
    dicoms, rois = find_inputs(case_dir)
    frames, times, part_idx, masks, labels, files = [], [], [], [], [], []
    offset = 0.0
    for i, d in enumerate(dicoms):
        f, t = load_dicom(d)
        if frames and f.shape[1:] != frames[0].shape[1:]:
            raise ValueError(f"{d.name}: image size differs from the previous part")
        num = part_number(d, case_dir)
        roi = pick_roi(rois, num, case_dir)
        masks.append(load_roi_mask(roi, f.shape[1:]))
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        t_shift = t - t[0] + offset
        offset = t_shift[-1] + dt
        frames.append(f)
        times.append(t_shift)
        part_idx.append(np.full(len(f), i))
        labels.append(num or 1)
        files.append((str(d.relative_to(case_dir)), str(roi.relative_to(case_dir))))
    return dict(frames=np.concatenate(frames), times=np.concatenate(times),
                part=np.concatenate(part_idx), masks=np.stack(masks),
                labels=np.array(labels), files=files)


def case_out(case_name):
    p = config.OUTPUT_DIR / case_name
    p.mkdir(parents=True, exist_ok=True)
    return p
