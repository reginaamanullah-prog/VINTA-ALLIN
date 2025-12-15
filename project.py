import streamlit as st
import numpy as np
import cv2
from PIL import Image
from math import radians, sin, cos
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import tempfile
import os

# ========== THEME CONTROLLER ==========
theme = st.sidebar.selectbox("🎨 Choose Theme", ["💗 Pink", "🌙 Dark Mode", "☀ Light Mode"])


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

# ========== NAVIGATION ==========
page = st.sidebar.radio(
    "📍 Navigate",
    ["🏠 Home", "🖼 Image Processing", "✂ Background Removal", "👥 Team", "📄 Report"],
)

# ========== SESSION STATE INIT ==========
if "original" not in st.session_state:
    st.session_state.original = None
if "processed" not in st.session_state:
    st.session_state.processed = None
if "team" not in st.session_state:
    st.session_state.team = ["Member 1", "Member 2"]

# ========== TRANSLATION DICT ==========
t = {
    "team_title": "👥 Team",
    "team_subtitle": "Meet our awesome team members!",
    "team_sid": "Student ID:",
    "team_role": "Role:",
    "team_group": "Group:",
    "team_contribution": "Contribution:"
}

def safe_display_square_image(img_path):
    try:
        img = Image.open(img_path)
        img = img.resize((150, 150))
        st.image(img)
    except FileNotFoundError:
        st.write("Image not found")
    except Exception as e:
        st.write(f"Error loading image: {e}")

# ========== MATRIX FUNCTIONS ==========
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
    M2 = shift @ M
    return cv2.warpAffine(img, M2, (new_w, new_h))

# ========== HOME ==========
if page == "🏠 Home":
    st.markdown(
        """
        <div style='text-align:center;padding:20px'>
        <h1>✨ Matrix Image Processing & Computer Vision ✨</h1>
        <h3>Transform images, remove background, generate PDF — All in one place 🎓</h3>
        <p>Choose theme on the sidebar 🎨 and start exploring features!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.info("🛠 **Transform Images** with matrix math")
    col2.info("🎨 **Edit & Filter** using convolution")
    col3.info("📄 **Download PDF Report** automatically")

    st.markdown("---")
    st.success("Use the sidebar to navigate ➡")

# ========== IMAGE PROCESSING ==========
elif page == "🖼 Image Processing":
    st.header("🖼 Image Processing Tools")
    file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if file is not None:
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)
        st.session_state.original = img
        st.image(img, caption="Original Image")

        action = st.selectbox("Select Tool", ["Matrix Transform", "Convolution Filter"])

        if action == "Matrix Transform":
            method = st.selectbox(
                "Transformation",
                ["Translation", "Scaling", "Rotation", "Shearing", "Reflection"],
            )

            if method == "Translation":
                tx = st.slider("Move X", -200, 200, 50)
                ty = st.slider("Move Y", -200, 200, 30)
                M = translation(tx, ty)
            elif method == "Scaling":
                sx = st.slider("Scale X", 0.1, 3.0, 1.2)
                sy = st.slider("Scale Y", 0.1, 3.0, 1.2)
                M = scaling(sx, sy)
            elif method == "Rotation":
                ang = st.slider("Angle", -180, 180, 45)
                h, w = img_np.shape[:2]
                M = rotation(ang, w / 2, h / 2)
            elif method == "Shearing":
                shx = st.slider("Shear X", -1.0, 1.0, 0.3)
                shy = st.slider("Shear Y", -1.0, 1.0, 0.0)
                M = shearing(shx, shy)
            else:
                axis = st.selectbox("Axis", ["x", "y"])
                M = reflection(axis)

            if st.button("Apply"):
                out = apply_transform(img_np, M)
                st.session_state.processed = Image.fromarray(out)
                st.image(out, caption="Transformed Result 🎉")

        else:
            filt = st.selectbox("Select Filter", ["Blur", "Sharpen"])
            if filt == "Blur":
                kernel = np.ones((3, 3), dtype=np.float32) / 9.0
            else:
                kernel = np.array(
                    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
                )
            if st.button("Apply Filter"):
                out = cv2.filter2D(img_np, -1, kernel)
                st.session_state.processed = Image.fromarray(out)
                st.image(out, caption="Filtered Result 🎉")

# ========== BACKGROUND REMOVAL ==========
elif page == "✂ Background Removal":
    st.header("✂ Background Removal")
    file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"], key="bg")

    if file is not None:
        img = np.array(Image.open(file).convert("RGB"))
        h, w = img.shape[:2]
        x = st.slider("X", 0, w, int(w * 0.1))
        y = st.slider("Y", 0, h, int(h * 0.1))
        rw = st.slider("Width", 10, w, int(w * 0.8))
        rh = st.slider("Height", 10, h, int(h * 0.8))

        if st.button("Remove Background"):
            mask = np.zeros((h, w), np.uint8)
            bgmodel = np.zeros((1, 65), np.float64)
            fgmodel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img, mask, (x, y, rw, rh), bgmodel, fgmodel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
            fg = img * mask2[:, :, np.newaxis]
            st.session_state.processed = Image.fromarray(fg)
            st.image(fg, caption="Background Removed 🎉")

elif page == "👥 Team":
    st.markdown(t["team_title"])
    st.write(t["team_subtitle"])


    members = [
        {"img": "images/Elizabeth.jpg", "name": "ELIZABETH KURNIAWAN", "sid": "04202400001", "role": "Leader", "Contribution": "Project Manager, Geometric Transformations Module"},
        {"img": "images/Regina.jpg", "name": "REGINA VINTA AMANULLAH", "sid": "04202400133", "role": "Member", "Contribution": "Image Filtering Module, UI/UX Design"},
        {"img": "images/Bill.jpg", "name": "BILL CHRISTIAN", "sid": "04202400058", "role": "Member", "Contribution": "Background Removal Module, Image Upload & Download"},
        {"img": "images/Putri.jpg", "name": "PUTRI LASRIDA MALAU", "sid": "04202400132", "role": "Member", "Contribution": "Histogram Module, Image Processing Functions"},
    ]

    cols_row1 = st.columns(2, vertical_alignment="top")
    for i in range(2):
        with cols_row1[i]:
            with st.container(border=True):
                m = members[i]
                _, col_img, _ = st.columns([1, 1, 1])
                with col_img:
                    safe_display_square_image(m["img"])
                st.markdown(f"{m['name']}")
                st.markdown(f"{t['team_sid']} {m['sid']}")
                st.markdown(f"{t['team_role']} {m['role']}")
                st.markdown(f"{t['team_group']} 5")
                st.markdown(f"{t['team_contribution']} {m['Contribution']}")

    cols_row2 = st.columns(2, vertical_alignment="top")
    for i in range(2, 4):
        with cols_row2[i - 2]:
            with st.container(border=True):
                m = members[i]
                _, col_img, _ = st.columns([1, 1, 1])
                with col_img:
                    safe_display_square_image(m["img"])
                st.markdown(f"{m['name']}")
                st.markdown(f"{t['team_sid']} {m['sid']}")
                st.markdown(f"{t['team_role']} {m['role']}")
                st.markdown(f"{t['team_group']} 5")
                st.markdown(f"{t['team_contribution']} {m['Contribution']}")

# ========== REPORT ==========
elif page == "📄 Report":
    st.header("📄 Generate PDF Report")
    title = st.text_input("Report Title", value="Matrix Image Processing Report")

    if st.button("Create PDF"):
        if st.session_state.original is None or st.session_state.processed is None:
            st.error("❗ Process an image before generating report")
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
                st.download_button("Download Report", f, file_name="report.pdf")

            os.remove(orig_path)
            os.remove(proc_path)
            st.success("🎉 PDF Created Successfully!")
