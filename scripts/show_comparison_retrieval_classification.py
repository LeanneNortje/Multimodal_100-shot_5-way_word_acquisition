import os
import streamlit as st

from pathlib import Path
from toolz import partition_all

from predict import COCOData
from evaluate import COCOResults

st.set_page_config(layout="wide")

TO_SAVE_DATA_FOR_PAPER = False
OUT_FOLDER = Path("output/taslp")

config_name_ret = "100-loc-v2-ret"
config_name_clf = "100-loc-v2-clf"

dataset = COCOData()
concepts = dataset.load_concepts()

@st.cache_data()
def load_results():
    results_ret = COCOResults(config_name_ret, dataset)
    results_clf = COCOResults(config_name_clf, dataset)
    return results_ret, results_clf

results_ret, results_clf = load_results()

with st.sidebar:
    query_concept = st.selectbox("query concept", concepts)
    episode_no = st.number_input(
        "episode no.", min_value=0, max_value=1000, format="%d", step=1
    )


# audio_query, _ = results100.episodes[episode_no]["queries"][query_concept]
# audio_path = AUDIO_COCO_DIR / audio_query
# audio_name = audio_path.stem

data = results_ret.load(query_concept, episode_no)
data = sorted(data, reverse=True, key=lambda datum: datum["score"])

for rank, datum in enumerate(data, start=1):
    datum["rank"] = rank

data_clf = results_clf.load(query_concept, episode_no)

TOP_K = 5
data = data[:TOP_K]
cols = st.columns(TOP_K)

for col, datum in zip(cols, data):
    image_file = datum["image-file"]
    image_path = dataset.image_dir / image_file
    image_name = image_path.stem

    col.markdown("rank: {:d} · score: {:.1f}".format(datum["rank"], datum["score"]))
    col.image(str(image_path))

cols = st.columns(TOP_K)

for col, datum in zip(cols, data_clf):
    image_file = datum["image-file"]
    image_path = dataset.image_dir / image_file
    image_name = image_path.stem

    col.markdown(
        "category: {:s} · score: {:.1f}".format(datum["image-concept"], datum["score"])
    )
    col.image(str(image_path))
    col.code(image_name)

    if TO_SAVE_DATA_FOR_PAPER:
        import shutil

        concept_str = query_concept.replace(" ", "-")
        shutil.copy(
            image_path,
            OUT_FOLDER
            / "imgs"
            / f"{config_name_clf}-{concept_str}-{episode_no}"
            / f"{image_name}.jpg",
        )
