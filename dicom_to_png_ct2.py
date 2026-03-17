#!/usr/bin/env python3
"""
DICOM series folder -> isotropic axial/coronal/sagittal image stacks (PNG/JPG)

Converted and parameterized from the notebook: "DICOM to PNG CT-2.ipynb"

Usage (CLI):
  python dicom_to_png_ct2.py --src "/path/to/dicom_folder" --out "/path/to/output" --fmt png --wl 40 --ww 400

This script is designed to be callable from Streamlit (import + function call) or from CLI.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from PIL import Image
from scipy.ndimage import zoom


def _is_dicom(p: Path) -> bool:
    """Fast DICOM check by looking for 'DICM' magic at byte 128."""
    try:
        with open(p, "rb") as f:
            pre = f.read(132)
            return pre[128:132] == b"DICM"
    except Exception:
        return False


def _safe_spacing(ds) -> tuple[float, float, float]:
    """
    Return spacing (px, py, pz) in mm.
    X=columns, Y=rows, Z=slice direction.
    """
    px = py = 1.0
    ps = getattr(ds, "PixelSpacing", None)
    if ps is not None and len(ps) >= 2:
        try:
            px, py = float(ps[0]), float(ps[1])
        except Exception:
            px = py = 1.0

    pz = getattr(ds, "SpacingBetweenSlices", None)
    if pz is not None:
        try:
            pz = float(pz)
        except Exception:
            pz = None
    if pz is None:
        try:
            pz = float(getattr(ds, "SliceThickness"))
        except Exception:
            pz = px

    return float(px), float(py), float(pz)


def _save_stack(stack3d: np.ndarray, out_dir: Path, fmt: str, flip_lr: bool = False, flip_ud: bool = False) -> None:
    """Save a 3D stack (N,H,W) as numbered grayscale images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    num_width = max(3, len(str(stack3d.shape[0])))
    fmt = fmt.lower().strip(".")
    for i, slice2d in enumerate(stack3d, start=1):  # 001-based
        img = (slice2d * 255.0 + 0.5).astype(np.uint8)
        if flip_lr:
            img = np.fliplr(img)
        if flip_ud:
            img = np.flipud(img)
        fname = f"{i:0{num_width}d}"
        if fmt in ("jpg", "jpeg"):
            Image.fromarray(img, mode="L").save(out_dir / f"{fname}.jpg", quality=95, subsampling=0)
        else:
            Image.fromarray(img, mode="L").save(out_dir / f"{fname}.png")


def convert_dicom_series_to_images(
    src_dir: str | Path,
    out_dir: str | Path,
    fmt: str = "png",
    wl: float | None = 40,
    ww: float | None = 400,
    auto_window: bool = False,
    iso_spacing_mm: float | None = None,
    interp_order: int = 1,
    save_axial: bool = True,
    save_coronal: bool = True,
    save_sagittal: bool = True,
    flip_axial_lr: bool = False,
    flip_axial_ud: bool = False,
    flip_cor_lr: bool = False,
    flip_cor_ud: bool = False,
    flip_sag_lr: bool = False,
    flip_sag_ud: bool = False,
) -> dict:
    """
    Convert DICOM series (pick the series with maximum slices under src_dir) to isotropic 3-plane image stacks.

    Returns a dict with summary metadata.
    """
    src = Path(src_dir).expanduser()
    out_root = Path(out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Collect DICOM files and group by (StudyUID, SeriesUID)
    series: dict[tuple[str | None, str | None], list[Path]] = {}
    for p in src.rglob("*"):
        if p.is_file() and _is_dicom(p):
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                key = (getattr(ds, "StudyInstanceUID", None), getattr(ds, "SeriesInstanceUID", None))
                if key[1]:
                    series.setdefault(key, []).append(p)
            except Exception:
                pass

    if not series:
        raise RuntimeError(f"No DICOM series found under: {src}")

    key = max(series, key=lambda k: len(series[k]))
    files = series[key]

    # 2) Sort by ImagePositionPatient(Z) if available else InstanceNumber
    metas = []
    for p in files:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        inst = int(getattr(ds, "InstanceNumber", 0))
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = float(ipp[2]) if (ipp and len(ipp) == 3) else None
        metas.append((p, z, inst))

    if any(m[1] is not None for m in metas):
        metas.sort(key=lambda x: (x[1] is None, x[1]))
    else:
        metas.sort(key=lambda x: x[2])

    files = [m[0] for m in metas]

    # 3) Load and HU conversion
    ds0 = pydicom.dcmread(str(files[0]), force=True)
    px, py, pz = _safe_spacing(ds0)  # mm
    Y, X = int(ds0.Rows), int(ds0.Columns)
    Z = len(files)

    vol = np.zeros((Z, Y, X), dtype=np.float32)
    for i, p in enumerate(files):
        ds = pydicom.dcmread(str(p), force=True)
        arr = ds.pixel_array
        try:
            arr = apply_modality_lut(arr, ds)
        except Exception:
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float32)

        if auto_window and hasattr(ds, "VOILUTSequence"):
            try:
                arr = apply_voi_lut(arr, ds).astype(np.float32)
            except Exception:
                pass

        vol[i] = arr

    # 4) Isotropic resampling (zoom factors in Z,Y,X)
    iso = min(px, py, pz) if iso_spacing_mm is None else float(iso_spacing_mm)

    fz, fy, fx = pz / iso, py / iso, px / iso
    newZ = max(1, int(round(Z * fz)))
    newY = max(1, int(round(Y * fy)))
    newX = max(1, int(round(X * fx)))

    vol_iso = zoom(vol, zoom=(fz, fy, fx), order=int(interp_order), mode="nearest", grid_mode=True)
    vol_iso = vol_iso[:newZ, :newY, :newX]  # guard small rounding drift

    # 5) Windowing
    if auto_window or wl is None or ww is None:
        Z2, Y2, X2 = vol_iso.shape
        samp = vol_iso[::max(Z2 // 16, 1), ::max(Y2 // 16, 1), ::max(X2 // 16, 1)].ravel()
        vmin, vmax = np.percentile(samp, 0.5), np.percentile(samp, 99.5)
    else:
        vmin, vmax = float(wl) - float(ww) / 2.0, float(wl) + float(ww) / 2.0

    vol01 = np.clip((vol_iso - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)

    # 6) Save 3 planes
    if save_axial:
        _save_stack(vol01, out_root / "axial", fmt=fmt, flip_lr=flip_axial_lr, flip_ud=flip_axial_ud)

    if save_coronal:
        cor = np.transpose(vol01, (1, 0, 2))  # (Y,Z,X)
        _save_stack(cor, out_root / "coronal", fmt=fmt, flip_lr=flip_cor_lr, flip_ud=flip_cor_ud)

    if save_sagittal:
        sag = np.transpose(vol01, (2, 0, 1))  # (X,Z,Y)
        _save_stack(sag, out_root / "sagittal", fmt=fmt, flip_lr=flip_sag_lr, flip_ud=flip_sag_ud)

    # 7) Metadata for downstream tools
    with open(out_root / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SeriesInstanceUID", key[1]])
        w.writerow(["Original_Shape_ZYX", Z, Y, X])
        w.writerow(["Original_Spacing_mm_XYZ", px, py, pz])
        w.writerow(["Isotropic_Spacing_mm", iso])
        Z2, Y2, X2 = vol01.shape
        w.writerow(["Isotropic_Shape_ZYX", Z2, Y2, X2])
        w.writerow(["Axial_pixel_mm_(H,W)_&slice_step_mm", (iso, iso), iso])
        w.writerow(["Coronal_pixel_mm_(H,W)_&slice_step_mm", (iso, iso), iso])
        w.writerow(["Sagittal_pixel_mm_(H,W)_&slice_step_mm", (iso, iso), iso])

    return {
        "series_uid": key[1],
        "original_shape_zyx": (int(Z), int(Y), int(X)),
        "original_spacing_mm_xyz": (float(px), float(py), float(pz)),
        "iso_spacing_mm": float(iso),
        "iso_shape_zyx": tuple(int(x) for x in vol01.shape),
        "out_dir": str(out_root),
        "fmt": fmt,
        "window": {"auto": bool(auto_window), "vmin": float(vmin), "vmax": float(vmax), "wl": wl, "ww": ww},
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert DICOM series to isotropic axial/coronal/sagittal PNG/JPG stacks.")
    p.add_argument("--src", required=True, help="Input folder containing DICOM series (recursive search).")
    p.add_argument("--out", required=True, help="Output folder (will create axial/coronal/sagittal subfolders).")
    p.add_argument("--fmt", default="png", choices=["png", "jpg", "jpeg"], help="Output image format.")
    p.add_argument("--wl", type=float, default=40.0, help="Window level (ignored if --auto).")
    p.add_argument("--ww", type=float, default=400.0, help="Window width (ignored if --auto).")
    p.add_argument("--auto", action="store_true", help="Use automatic windowing (0.5–99.5 percentile).")
    p.add_argument("--iso", type=float, default=None, help="Isotropic spacing in mm. Default: min(px,py,pz).")
    p.add_argument("--interp", type=int, default=1, choices=[0,1,3], help="Interpolation order for resampling.")
    p.add_argument("--no-axial", action="store_true")
    p.add_argument("--no-coronal", action="store_true")
    p.add_argument("--no-sagittal", action="store_true")
    return p


def main() -> None:
    p = _build_parser()
    args = p.parse_args()

    info = convert_dicom_series_to_images(
        src_dir=args.src,
        out_dir=args.out,
        fmt=args.fmt,
        wl=args.wl,
        ww=args.ww,
        auto_window=args.auto,
        iso_spacing_mm=args.iso,
        interp_order=args.interp,
        save_axial=not args.no_axial,
        save_coronal=not args.no_coronal,
        save_sagittal=not args.no_sagittal,
    )

    print("✅ Done.")
    print("Series:", info["series_uid"])
    print("Original  (Z,Y,X):", info["original_shape_zyx"], "Spacing(mm XYZ):", info["original_spacing_mm_xyz"])
    print("Isotropic (Z,Y,X):", info["iso_shape_zyx"], "Spacing(mm):", info["iso_spacing_mm"])
    print("Saved to:", info["out_dir"])


if __name__ == "__main__":
    main()
