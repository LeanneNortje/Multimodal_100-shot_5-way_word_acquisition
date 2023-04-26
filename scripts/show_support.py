import numpy as np
import streamlit as st

from predict import IMAGE_COCO_DIR
st.set_page_config(layout="wide")


path = "support_set/support_set_100.npz"
support = np.load(path, allow_pickle=True)["support_set"].item()

for key, value in support.items():
    _, image_path, *_, keyword = value
    image_path = IMAGE_COCO_DIR / image_path
    st.markdown(keyword)
    st.image(str(image_path))
    st.markdown("---")
