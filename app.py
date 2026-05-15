import streamlit as st
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

st.header("Matoux Segmentation")

IMAGES_DIR = Path(__file__).parent / "images"
SUPPORTED = (".png", ".jpg", ".jpeg")

if not IMAGES_DIR.exists():
    st.error(f"Directory `{IMAGES_DIR}` not found.")
    st.stop()

images = sorted([p.name for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED])

if not images:
    st.warning(f"No images found in `{IMAGES_DIR}`.")
    st.stop()

selected_images = st.multiselect("Select images", images)

if not selected_images:
    st.info("No image selected.")
    st.stop()

cols = st.columns(6)
for i, name in enumerate(selected_images):
    image_path = IMAGES_DIR / name
    try:
        image = Image.open(image_path)
        cols[i].image(image, caption=Path(name).stem, width="stretch")
    except Exception as e:
        cols[i].error(f"Failed to open image `{image_path}`: {e}")


st.subheader("Color Space Conversion")
color_options = ["RGB", "HSV", "LAB", "YCbCr"]
selected_colors = st.multiselect("Select color spaces", color_options)
num_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=7, value=3)
extract_contour = st.toggle("Extract Contour", value=True, key="color_space_contour")

if selected_colors:
    cols = st.columns(len(selected_colors) + 1)
    cols[0].caption("Original", text_alignment="center")
    for i, color in enumerate(selected_colors):
        cols[i + 1].caption(color, text_alignment="center")

    for i, name in enumerate(selected_images):
        image_path = IMAGES_DIR / name
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("cv2.imread returned None")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).shape[:2]
            cy, cx = (height // 2, width // 2)
            crop_size = round(min(height, width) * 0.1)

            cols = st.columns(len(selected_colors) + 1)
            cols[0].image(image_rgb, width="stretch")
            for j, color in enumerate(selected_colors):
                if color == "HSV":
                    image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                elif color == "LAB":
                    image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                elif color == "YCbCr":
                    image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                else:
                    image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image_converted = image_converted.reshape((-1, 3)).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                retval, labels, centers = cv2.kmeans(image_converted, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                labels = labels.reshape(height, width)
                crop = labels[cy - crop_size:cy + crop_size, cx - crop_size:cx + crop_size]
                center_cluster = np.bincount(crop.flatten()).argmax()
                image_masked = (labels == center_cluster).astype(np.uint8) * 255

                if extract_contour:
                    contours, _ = cv2.findContours(image_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    image_masked = np.zeros_like(image_masked)
                    if contours:
                        best_contour = max(contours, key=cv2.contourArea)
                        cv2.drawContours(image_masked, [best_contour], -1, 255, cv2.FILLED)
                else:
                    image_masked = cv2.cvtColor(image_masked, cv2.COLOR_GRAY2BGR)

                cols[j + 1].image(image_masked, width="stretch")
        except Exception as e:
            st.error(f"Failed to process `{image_path}`: {e}")
else:
    st.info("Select at least one color space to compare.")


st.subheader("K-Means Clustering")
cluster_options = [2, 3, 4, 5, 6, 7]
selected_clusters = st.multiselect("Select cluster counts (k)", cluster_options)
color_options = ["RGB", "HSV", "LAB", "YCbCr"]
selected_color = st.selectbox("Select a color space", color_options)
extract_contour = st.toggle("Extract Contour", value=True, key="kmeans_contour")

if selected_clusters:
    cols = st.columns(len(selected_clusters) + 1)
    cols[0].caption("Original", text_alignment="center")
    for i, cluster in enumerate(selected_clusters):
        cols[i + 1].caption(f"k={cluster}", text_alignment="center")

    for i, name in enumerate(selected_images):
        image_path = IMAGES_DIR / name
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("cv2.imread returned None")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).shape[:2]
            cy, cx = (height // 2, width // 2)
            crop_size = round(min(height, width) * 0.1)

            cols = st.columns(len(selected_clusters) + 1)
            cols[0].image(image_rgb, width="stretch")
            if selected_color == "HSV":
                image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif selected_color == "LAB":
                image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            elif selected_color == "YCbCr":
                image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            else:
                image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_converted = image_converted.reshape((-1, 3)).astype(np.float32)

            for j, num_clusters in enumerate(selected_clusters):
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                retval, labels, centers = cv2.kmeans(image_converted, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                labels = labels.reshape(height, width)
                crop = labels[cy - crop_size:cy + crop_size, cx - crop_size:cx + crop_size]
                center_cluster = np.bincount(crop.flatten()).argmax()
                image_masked = (labels == center_cluster).astype(np.uint8) * 255

                if extract_contour:
                    contours, _ = cv2.findContours(image_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    image_masked = np.zeros_like(image_masked)
                    if contours:
                        best_contour = max(contours, key=cv2.contourArea)
                        cv2.drawContours(image_masked, [best_contour], -1, 255, cv2.FILLED)
                else:
                    image_masked = cv2.cvtColor(image_masked, cv2.COLOR_GRAY2BGR)

                cols[j + 1].image(image_masked, width="stretch")
        except Exception as e:
            st.error(f"Failed to process `{image_path}`: {e}")
else:
    st.info("Select at least one cluster count to compare.")


st.subheader("Color Channel Extraction")
channel_options = ["R/H/L/Y", "G/S/A/Cr", "B/V/B/Cb"]
selected_channels = st.multiselect("Select color channel", channel_options)
color_options = ["RGB", "HSV", "LAB", "YCbCr"]
selected_color = st.selectbox("Select a color space", color_options, key="color_extraction_color")
num_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=7, value=3, key="color_extraction_num_clusters")
extract_contour = st.toggle("Extract Contour", value=True, key="color_extraction_contour")

if selected_channels:
    cols = st.columns(len(selected_channels) + 1)
    cols[0].caption("Original", text_alignment="center")
    for i, channel in enumerate(selected_channels):
        cols[i + 1].caption(channel, text_alignment="center")

    for i, name in enumerate(selected_images):
        image_path = IMAGES_DIR / name
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("cv2.imread returned None")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).shape[:2]
            cy, cx = (height // 2, width // 2)
            crop_size = round(min(height, width) * 0.1)

            cols = st.columns(len(selected_channels) + 1)
            cols[0].image(image_rgb, width="stretch")
            if selected_color == "HSV":
                image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif selected_color == "LAB":
                image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            elif selected_color == "YCbCr":
                image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            else:
                image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rhly, gsacr, bvbcr = cv2.split(image_converted)

            for j, selected_channel in enumerate(selected_channels):
                if selected_channel == "G/S/A/Cr":
                    image_converted = gsacr.reshape((-1, 1)).astype(np.float32)
                elif selected_channel == "B/V/B/Cb":
                    image_converted = bvbcr.reshape((-1, 1)).astype(np.float32)
                else:
                    image_converted = rhly.reshape((-1, 1)).astype(np.float32)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                retval, labels, centers = cv2.kmeans(image_converted, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                labels = labels.reshape(height, width)
                crop = labels[cy - crop_size:cy + crop_size, cx - crop_size:cx + crop_size]
                center_cluster = np.bincount(crop.flatten()).argmax()
                image_masked = (labels == center_cluster).astype(np.uint8) * 255

                if extract_contour:
                    contours, _ = cv2.findContours(image_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    image_masked = np.zeros_like(image_masked)
                    if contours:
                        best_contour = max(contours, key=cv2.contourArea)
                        cv2.drawContours(image_masked, [best_contour], -1, 255, cv2.FILLED)
                else:
                    image_masked = cv2.cvtColor(image_masked, cv2.COLOR_GRAY2BGR)

                cols[j + 1].image(image_masked, width="stretch")
        except Exception as e:
            st.error(f"Failed to process `{image_path}`: {e}")
else:
    st.info("Select at least one color channel to compare.")
