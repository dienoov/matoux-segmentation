import cv2
import numpy as np
import streamlit as st


def k_means_clustering(images):
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

        for i, image in enumerate(images):
            try:
                image = cv2.imread(str(image))
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
                    retval, labels, centers = cv2.kmeans(
                        image_converted,
                        num_clusters,
                        None,
                        criteria,
                        10,
                        cv2.KMEANS_RANDOM_CENTERS,
                    )

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
                st.error(f"Failed to process `{image}`: {e}")
    else:
        st.info("Select at least one cluster count to compare.")
