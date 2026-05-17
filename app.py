from pathlib import Path

import streamlit as st
from PIL import Image

from color_channel_extraction import color_channel_extraction
from color_space_conversion import color_space_conversion
from k_means_clustering import k_means_clustering

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

cols = st.columns(len(selected_images))
for i, name in enumerate(selected_images):
    image_path = IMAGES_DIR / name
    try:
        image = Image.open(image_path)
        cols[i].image(image, caption=Path(name).stem, width="stretch")
    except Exception as e:
        cols[i].error(f"Failed to open image `{image_path}`: {e}")

paths = [IMAGES_DIR / name for name in selected_images]

color_space_conversion(paths)
k_means_clustering(paths)
color_channel_extraction(paths)
