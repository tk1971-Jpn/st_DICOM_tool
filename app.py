import streamlit as st
from pathlib import Path
from datetime import datetime

from dicom_to_png_ct2 import convert_dicom_series_to_images

st.set_page_config(page_title="DICOM → PNG/JPG (Local)", layout="centered")
st.title("DICOM → PNG/JPG（ローカル変換ツール）")

st.caption("ローカルPC上で完結。DICOMフォルダを指定して、Axial/Coronal/Sagittalの画像スタックを出力します。")

# ---- Inputs ----
src = st.text_input("DICOMフォルダ（入力）", value=str(Path("~/Desktop/DICOM").expanduser()))
out_root = st.text_input("出力フォルダ（親）", value=str(Path("~/Desktop/out_iso_views").expanduser()))

col1, col2, col3 = st.columns(3)
with col1:
    fmt = st.selectbox("形式", ["png", "jpg"], index=0)
with col2:
    wl = st.number_input("WL（Window Level）", value=40.0)
with col3:
    ww = st.number_input("WW（Window Width）", value=400.0)

auto = st.checkbox("自動WL/WW（0.5〜99.5%）", value=False)

iso_spacing_mm = st.text_input("等方化 spacing(mm)（空欄なら最小px/py/pz）", value="")
interp_order = st.selectbox("補間（等方化）", [0, 1, 3], index=1, help="0=Nearest, 1=Linear, 3=Cubic")

st.markdown("### 出力方向")
c1, c2, c3 = st.columns(3)
with c1:
    save_axial = st.checkbox("Axial", value=True)
with c2:
    save_coronal = st.checkbox("Coronal", value=True)
with c3:
    save_sagittal = st.checkbox("Sagittal", value=True)

st.markdown("### 反転（必要な場合のみ）")
f1, f2, f3 = st.columns(3)
with f1:
    flip_axial_lr = st.checkbox("Axial LR", value=False)
    flip_axial_ud = st.checkbox("Axial UD", value=False)
with f2:
    flip_cor_lr = st.checkbox("Coronal LR", value=False)
    flip_cor_ud = st.checkbox("Coronal UD", value=False)
with f3:
    flip_sag_lr = st.checkbox("Sagittal LR", value=False)
    flip_sag_ud = st.checkbox("Sagittal UD", value=False)

run = st.button("変換を実行", type="primary")

if run:
    src_p = Path(src).expanduser()
    out_root_p = Path(out_root).expanduser()
    if not src_p.exists():
        st.error(f"入力フォルダが存在しません: {src_p}")
        st.stop()
    out_root_p.mkdir(parents=True, exist_ok=True)

    # 上書き事故防止：日時サブフォルダへ
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root_p / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    iso = None
    if iso_spacing_mm.strip():
        try:
            iso = float(iso_spacing_mm.strip())
        except Exception:
            st.error("等方化 spacing(mm) は数値で入力してください（例: 1.0）")
            st.stop()

    with st.spinner("DICOMを読み込み、等方化して画像を書き出しています…"):
        try:
            info = convert_dicom_series_to_images(
                src_dir=src_p,
                out_dir=out_dir,
                fmt=fmt,
                wl=None if auto else wl,
                ww=None if auto else ww,
                auto_window=auto,
                iso_spacing_mm=iso,
                interp_order=int(interp_order),
                save_axial=save_axial,
                save_coronal=save_coronal,
                save_sagittal=save_sagittal,
                flip_axial_lr=flip_axial_lr,
                flip_axial_ud=flip_axial_ud,
                flip_cor_lr=flip_cor_lr,
                flip_cor_ud=flip_cor_ud,
                flip_sag_lr=flip_sag_lr,
                flip_sag_ud=flip_sag_ud,
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success("完了しました。")
    st.write("出力先:", info["out_dir"])
    st.json(info)

    # Simple preview: show first few axial images if present
    axial_dir = Path(info["out_dir"]) / "axial"
    if axial_dir.exists():
        imgs = sorted(axial_dir.glob("*.png")) + sorted(axial_dir.glob("*.jpg"))
        if imgs:
            st.markdown("### プレビュー（Axialの先頭数枚）")
            st.image([str(p) for p in imgs[:6]], caption=[p.name for p in imgs[:6]])
