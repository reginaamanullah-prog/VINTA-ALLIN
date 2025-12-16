import streamlit as st
import numpy as np
import cv2
from PIL import Image
from math import radians, sin, cos
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import tempfile
import os

# ================== LANGUAGE & THEME ==================
lang = st.sidebar.selectbox("🌐 Language / Bahasa", ["English", "Indonesia"])

theme = st.sidebar.selectbox("🎨 Theme", ["💗 Pink", "🌙 Dark Mode", "☀ Light Mode"])


def apply_theme():
    if theme == "💗 Pink":
        st.markdown(
            """
            <style>
            body { background-color: #ffe6f2; }
            .stApp { background-color: #ffe6f2; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif theme == "🌙 Dark Mode":
        st.markdown(
            """
            <style>
            body { background-color: #0e1117; color:white; }
            .stApp { background-color: #0e1117; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            body { background-color: #ffffff; }
            .stApp { background-color: #ffffff; }
            </style>
            """,
            unsafe_allow_html=True,
        )


apply_theme()

# ================== TRANSLATION DICT ==================
T = {
    "English": {
        "nav": ["🏠 Home", "🖼 Image Processing", "✂ Background Removal", "👥 Team", "📄 Report"],
        "home_title": "✨ Matrix Image Processing & Computer Vision ✨",
        "home_subtitle": "Transform images, remove background, generate PDF — All in one place 🎓",
        "home_hint": "Choose theme on the sidebar 🎨 and start exploring features!",
        "home_box1": "🛠 Transform Images with matrix math",
        "home_box2": "🎨 Edit & Filter using convolution",
        "home_box3": "📄 Download PDF Report automatically",
        "home_foot": "Use the sidebar to navigate ➡",

        "img_header": "🖼 Image Processing Tools",
        "img_upload": "Upload Image",
        "img_tool": "Select Tool",
        "img_tool_opts": ["Matrix Transform", "Convolution Filter"],
        "transform_label": "Transformation",
        "transform_opts": ["Translation", "Scaling", "Rotation", "Shearing", "Reflection"],
        "translation_tx": "Move X",
        "translation_ty": "Move Y",
        "scaling_sx": "Scale X",
        "scaling_sy": "Scale Y",
        "rotation_ang": "Angle",
        "shear_x": "Shear X",
        "shear_y": "Shear Y",
        "reflection_axis": "Axis",
        "reflection_opts": ["x", "y"],
        "btn_apply": "Apply",
        "conv_filter": "Select Filter",
        "conv_opts": ["Blur", "Sharpen"],

        "bg_header": "✂ Background Removal",
        "bg_upload": "Upload image",
        "bg_x": "X",
        "bg_y": "Y",
        "bg_w": "Width",
        "bg_h": "Height",
        "bg_btn": "Remove Background",

        "team_title": "👥 Team",
        "team_subtitle": "Meet our awesome team members!",
        "team_sid": "Student ID:",
        "team_role": "Role:",
        "team_group": "Group:",
        "team_contribution": "Contribution:",

        "report_header": "📄 Generate PDF Report",
        "report_title": "Report Title",
        "report_default": "Matrix Image Processing Report",
        "report_btn": "Create PDF",
        "report_error": "❗ Process an image before generating report",
        "report_download": "Download Report",
        "report_success": "🎉 PDF Created Successfully!",

        "orig_caption": "Original Image",
        "transformed_caption": "Transformed Result 🎉",
        "filtered_caption": "Filtered Result 🎉",
        "bg_removed_caption": "Background Removed 🎉",
    },
    "Indonesia": {
        "nav": ["🏠 Beranda", "🖼 Pemrosesan Gambar", "✂ Hapus Background", "👥 Tim", "📄 Laporan"],
        "home_title": "✨ Pemrosesan Citra Matriks & Computer Vision ✨",
        "home_subtitle": "Transformasi gambar, hapus background, buat PDF — Semua dalam satu aplikasi 🎓",
        "home_hint": "Pilih tema di sidebar 🎨 dan mulai eksplor fitur!",
        "home_box1": "🛠 Transformasi Gambar dengan matriks",
        "home_box2": "🎨 Edit & Filter dengan konvolusi",
        "home_box3": "📄 Unduh Laporan PDF secara otomatis",
        "home_foot": "Gunakan sidebar untuk navigasi ➡",

        "img_header": "🖼 Tools Pemrosesan Gambar",
        "img_upload": "Upload Gambar",
        "img_tool": "Pilih Tool",
        "img_tool_opts": ["Transformasi Matriks", "Filter Konvolusi"],
        "transform_label": "Jenis Transformasi",
        "transform_opts": ["Translasi", "Skala", "Rotasi", "Shearing", "Refleksi"],
        "translation_tx": "Geser X",
        "translation_ty": "Geser Y",
        "scaling_sx": "Skala X",
        "scaling_sy": "Skala Y",
        "rotation_ang": "Sudut",
        "shear_x": "Shear X",
        "shear_y": "Shear Y",
        "reflection_axis": "Sumbu",
        "reflection_opts": ["x", "y"],
        "btn_apply": "Terapkan",
        "conv_filter": "Pilih Filter",
        "conv_opts": ["Blur", "Tajamkan"],

        "bg_header": "✂ Hapus Background",
        "bg_upload": "Upload gambar",
        "bg_x": "X",
        "bg_y": "Y",
        "bg_w": "Lebar",
        "bg_h": "Tinggi",
        "bg_btn": "Hapus Background",

        "team_title": "👥 Tim",
        "team_subtitle": "Kenalan dengan anggota tim kami!",
        "team_sid": "NIM:",
        "team_role": "Peran:",
        "team_group": "Kelompok:",
        "team_contribution": "Kontribusi:",

        "report_header": "📄 Buat Laporan PDF",
        "report_title": "Judul Laporan",
        "report_default": "Laporan Pemrosesan Citra Matriks",
        "report_btn": "Buat PDF",
        "report_error": "❗ Proses gambar terlebih dahulu sebelum membuat laporan",
        "report_download": "Unduh Laporan",
        "report_success": "🎉 PDF Berhasil Dibuat!",

        "orig_caption": "Gambar Asli",
        "transformed_caption": "Hasil Transformasi 🎉",
        "filtered_caption": "Hasil Filter 🎉",
        "bg_removed_caption": "Background Terhapus 🎉",
    },
}

t = T[lang]

# ================== NAVIGATION ==================
page = st.sidebar.radio("📍 Navigate", t["nav"])

# ================== SESSION STATE ==================
if "original" not in st.session_state:
    st.session_state.original = None
if "processed" not in st.session_state:
    st.session_state.processed = None

# ================== UTIL: SAFE IMAGE ==================
def safe_display_square_image(img_path):
    try:
        img = Image.open(img_path)
        img = img.resize((150, 150))
        st.image(img)
    except FileNotFoundError:
        st.write("Image not found")
    except Exception as e:
        st.write(f"Error loading image: {e}")

# ================== MATRIX FUNCTIONS ==================
def translation(tx, ty):
    return np.float32([[1, 0, tx], [0, 1, ty]])

def scaling(sx, sy):
    return np.float32([[sx, 0, 0], [0, sy, 0]])

def rotation(angle, cx, cy):
    r = radians(angle)
    return np.float32(
        [
            [cos(r), -sin(r), cx - cos(r) * cx + sin(r) * cy],
            [sin(r), cos(r), cy - sin(r) * cx - cos(r) * cy],
        ]
    )

def shearing(shx, shy):
    return np.float32([[1, shx, 0], [shy, 1, 0]])

def reflection(axis):
    if axis == "x":
        return np.float32([[1, 0, 0], [0, -1, 0]])
    else:
        return np.float32([[-1, 0, 0], [0, 1, 0]])

def apply_transform(img, M):
    h, w = img.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    ones = np.ones((4, 1), dtype=np.float32)
    c = np.hstack([corners, ones])
    M3 = np.vstack([M, [0, 0, 1]])
    new = (M3 @ c.T).T
    min_x, max_x = int(new[:, 0].min()), int(new[:, 0].max())
    min_y, max_y = int(new[:, 1].min()), int(new[:, 1].max())
    new_w, new_h = max_x - min_x, max_y - min_y
    shift = np.float32([[1, 0, -min_x], [0, 1, -min_y]])
    shift_3x3 = np.vstack([shift, [0, 0, 1]])
    M_3x3 = np.vstack([M, [0, 0, 1]])
    combined = shift_3x3 @ M_3x3
    M2 = combined[:2, :]
    return cv2.warpAffine(img, M2, (new_w, new_h))

# ================== HOME ==================
if page == t["nav"][0]:
    st.markdown(
        f"""
        <div style='text-align:center;padding:20px'>
        <h1>{t["home_title"]}</h1>
        <h3>{t["home_subtitle"]}</h3>
        <p>{t["home_hint"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.info(t["home_box1"])
    col2.info(t["home_box2"])
    col3.info(t["home_box3"])

    st.markdown("---")
    st.success(t["home_foot"])

# ================== IMAGE PROCESSING ==================
elif page == t["nav"][1]:
    st.header(t["img_header"])
    file = st.file_uploader(t["img_upload"], type=["png", "jpg", "jpeg"])

    if file is not None:
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)
        st.session_state.original = img
        st.image(img, caption=t["orig_caption"])

        action = st.selectbox(t["img_tool"], t["img_tool_opts"])

        # ---- Matrix Transform ----
        if action == t["img_tool_opts"][0]:
            method = st.selectbox(t["transform_label"], t["transform_opts"])

            # Mapping label internasional supaya fungsi tetap benar
            if lang == "English":
                trans_translation = "Translation"
                trans_scaling = "Scaling"
                trans_rotation = "Rotation"
                trans_shearing = "Shearing"
                trans_reflection = "Reflection"
            else:
                trans_translation = "Translasi"
                trans_scaling = "Skala"
                trans_rotation = "Rotasi"
                trans_shearing = "Shearing"
                trans_reflection = "Refleksi"

            if method == trans_translation:
                tx = st.slider(t["translation_tx"], -200, 200, 50)
                ty = st.slider(t["translation_ty"], -200, 200, 30)
                M = translation(tx, ty)
            elif method == trans_scaling:
                sx = st.slider(t["scaling_sx"], 0.1, 3.0, 1.2)
                sy = st.slider(t["scaling_sy"], 0.1, 3.0, 1.2)
                M = scaling(sx, sy)
            elif method == trans_rotation:
                ang = st.slider(t["rotation_ang"], -180, 180, 45)
                h, w = img_np.shape[:2]
                M = rotation(ang, w / 2, h / 2)
            elif method == trans_shearing:
                shx = st.slider(t["shear_x"], -1.0, 1.0, 0.3)
                shy = st.slider(t["shear_y"], -1.0, 1.0, 0.0)
                M = shearing(shx, shy)
            else:
                axis = st.selectbox(t["reflection_axis"], t["reflection_opts"])
                M = reflection(axis)

            if st.button(t["btn_apply"]):
                out = apply_transform(img_np, M)
                st.session_state.processed = Image.fromarray(out)
                st.image(out, caption=t["transformed_caption"])

        # ---- Convolution Filter ----
        else:
            filt = st.selectbox(t["conv_filter"], t["conv_opts"])

            # label blur di dua bahasa boleh tetap "Blur"
            if lang == "English":
                blur_label = "Blur"
                sharpen_label = "Sharpen"
            else:
                blur_label = "Blur"
                sharpen_label = "Tajamkan"

            if filt == blur_label:
                kernel = np.ones((3, 3), dtype=np.float32) / 9.0
            else:
                kernel = np.array(
                    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
                )

            if st.button(t["btn_apply"]):
                out = cv2.filter2D(img_np, -1, kernel)
                st.session_state.processed = Image.fromarray(out)
                st.image(out, caption=t["filtered_caption"])

# ================== BACKGROUND REMOVAL ==================
elif page == t["nav"][2]:
    st.header(t["bg_header"])
    file = st.file_uploader(t["bg_upload"], type=["jpg", "png", "jpeg"], key="bg")

    if file is not None:
        img = np.array(Image.open(file).convert("RGB"))
        h, w = img.shape[:2]
        x = st.slider(t["bg_x"], 0, w, int(w * 0.1))
        y = st.slider(t["bg_y"], 0, h, int(h * 0.1))
        rw = st.slider(t["bg_w"], 10, w, int(w * 0.8))
        rh = st.slider(t["bg_h"], 10, h, int(h * 0.8))

        if st.button(t["bg_btn"]):
            mask = np.zeros((h, w), np.uint8)
            bgmodel = np.zeros((1, 65), np.float64)
            fgmodel = np.zeros((1, 65), np.float64)
            # GrabCut dengan rectangle seperti contoh di dokumentasi OpenCV.[web:16][web:13]
            cv2.grabCut(img, mask, (x, y, rw, rh), bgmodel, fgmodel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
            fg = img * mask2[:, :, np.newaxis]
            st.session_state.processed = Image.fromarray(fg)
            st.image(fg, caption=t["bg_removed_caption"])

# ================== TEAM ==================
elif page == t["nav"][3]:
    st.markdown(t["team_title"])
    st.write(t["team_subtitle"])

    members = [
        {"img": "images/Elizabeth.jpg", "name": "ELIZABETH KURNIAWAN", "sid": "04202400001", "role": "Leader", "Contribution": "Project Manager, Geometric Transformations Module"},
        {"img": "images/Regina.jpg", "name": "REGINA VINTA AMANULLAH", "sid": "04202400133", "role": "Member", "Contribution": "Image Filtering Module, UI/UX Design"},
        {"img": "images/Bill.jpg", "name": "BILL CHRISTIAN", "sid": "04202400058", "role": "Member", "Contribution": "Background Removal Module, Image Upload & Download"},
        {"img": "images/Putri.jpg", "name": "PUTRI LASRIDA MALAU", "sid": "04202400132", "role": "Member", "Contribution": "Histogram Module, Image Processing Functions"},
    ]

    cols_row1 = st.columns(2)
    for i in range(2):
        with cols_row1[i]:
            with st.container():  # tanpa border agar aman di semua versi.[web:1]
                _, col_img, _ = st.columns([1, 1, 1])
                m = members[i]
                with col_img:
                    safe_display_square_image(m["img"])
                st.markdown(f"{m['name']}")
                st.markdown(f"{t['team_sid']} {m['sid']}")
                st.markdown(f"{t['team_role']} {m['role']}")
                st.markdown(f"{t['team_group']} 5")
                st.markdown(f"{t['team_contribution']} {m['Contribution']}")

    cols_row2 = st.columns(2)
    for i in range(2, 4):
        with cols_row2[i - 2]:
            with st.container():
                _, col_img, _ = st.columns([1, 1, 1])
                m = members[i]
                with col_img:
                    safe_display_square_image(m["img"])
                st.markdown(f"{m['name']}")
                st.markdown(f"{t['team_sid']} {m['sid']}")
                st.markdown(f"{t['team_role']} {m['role']}")
                st.markdown(f"{t['team_group']} 5")
                st.markdown(f"{t['team_contribution']} {m['Contribution']}")

# ================== REPORT ==================
elif page == t["nav"][4]:
    st.header(t["report_header"])
    title = st.text_input(t["report_title"], value=t["report_default"])

    if st.button(t["report_btn"]):
        if st.session_state.original is None or st.session_state.processed is None:
            st.error(t["report_error"])
        else:
            path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
            c = canvas.Canvas(path, pagesize=A4)
            c.drawString(50, 800, title)

            orig = st.session_state.original
            proc = st.session_state.processed

            orig_path = "orig.png"
            proc_path = "proc.png"
            orig.save(orig_path)
            proc.save(proc_path)

            c.drawImage(orig_path, 50, 500, 200, 200)
            c.drawImage(proc_path, 300, 500, 200, 200)
            c.save()

            with open(path, "rb") as f:
                st.download_button(t["report_download"], f, file_name="report.pdf")

            os.remove(orig_path)
            st.success(t["report_success"])
