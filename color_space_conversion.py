import cv2
import numpy as np
import streamlit as st

COLOR_SPACE_CHANNELS = {
    "RGB": ["R", "G", "B"],
    "HSV": ["H", "S", "V"],
    "LAB": ["L", "A", "B"],
    "YCbCr": ["Y", "Cb", "Cr"],
}


def color_space_conversion(images):
    st.subheader("Color Space Conversion")
    color_options = ["RGB", "HSV", "LAB", "YCbCr"]
    selected_colors = st.multiselect("Select color spaces", color_options)
    num_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=7, value=3)
    channel_selections: dict[str, list[int]] = {}
    for color in selected_colors:
        channels = COLOR_SPACE_CHANNELS[color]
        cols = st.columns(len(channels) + 1, vertical_alignment="center")
        cols[0].text(color)
        selected_channels = []
        for i, ch in enumerate(channels):
            default = True
            key = f"ch_{color}_{ch}"
            if cols[i + 1].checkbox(ch, value=default, key=key):
                selected_channels.append(i)
        if not selected_channels:
            st.warning(f"Select at least one {color} channel.")
        channel_selections[color] = selected_channels
    extract_contour = st.toggle("Extract Contour", value=True, key="color_space_contour")

    if selected_colors:
        cols = st.columns(len(selected_colors) + 1)
        cols[0].caption("Original", text_alignment="center")
        for i, color in enumerate(selected_colors):
            active_chs = channel_selections.get(color, [0, 1, 2])
            ch_names = COLOR_SPACE_CHANNELS[color]
            label = f"{color} ({'+'.join(ch_names[c] for c in active_chs) if active_chs else '—'})"
            cols[i + 1].caption(label, text_alignment="center")

        for i, image in enumerate(images):
            try:
                image = cv2.imread(str(image))
                if image is None:
                    raise ValueError("cv2.imread returned None")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image_rgb.shape[:2]
                cy, cx = height // 2, width // 2
                crop_size = round(min(height, width) * 0.1)

                cols = st.columns(len(selected_colors) + 1)
                cols[0].image(image_rgb, width="stretch")

                for j, color in enumerate(selected_colors):
                    active_chs = channel_selections.get(color, [0, 1, 2])

                    if color == "HSV":
                        image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    elif color == "LAB":
                        image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    elif color == "YCbCr":
                        image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                    else:
                        image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    if active_chs:
                        image_for_kmeans = image_converted[:, :, active_chs]
                    else:
                        image_for_kmeans = image_converted

                    n_ch = image_for_kmeans.shape[2] if image_for_kmeans.ndim == 3 else 1
                    flat = image_for_kmeans.reshape((-1, n_ch)).astype(np.float32)

                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                    _, labels, _ = cv2.kmeans(
                        flat,
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
                        contours, _ = cv2.findContours(
                            image_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
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
        st.info("Select at least one color space to compare.")