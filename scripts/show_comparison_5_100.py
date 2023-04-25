import streamlit as st

from toolz import partition_all

from predict import (
    AUDIO_COCO_DIR,
    IMAGE_COCO_DIR,
    MattNet,
    load_alignments,
    load_captions,
    load_concepts,
)
from evaluate import Results

st.set_page_config(layout="wide")


@st.cache_data
def load_resources():
    return (
        load_concepts(),
        load_captions(),
        Results(5),
        Results(100),
    )


concepts, captions_data, results5, results100 = load_resources()
results = {
    5: results5,
    100: results100,
}


with st.sidebar:
    query_concept = st.selectbox("query concept", concepts)
    episode_no = st.number_input(
        "episode no.", min_value=0, max_value=1000, format="%d", step=1
    )
    to_show_only_concepts = st.checkbox("images contain one of the concepts")


# audio_query, _ = results100.episodes[episode_no]["queries"][query_concept]
# audio_path = AUDIO_COCO_DIR / audio_query
# audio_name = audio_path.stem


num_columns = 10
num_pos = {
    "broccoli": 57,
    "fire hydrant": 62,
    "kite": 91,
    "sheep": 63,
    "zebra": 90,
}

episodes = results[5].episodes["matching_set"]


for k in [5, 100]:
    data = results[k].load(query_concept, episode_no)
    data = sorted(data, reverse=True, key=lambda datum: datum["score"])

    for rank, datum in enumerate(data, start=1):
        datum["rank"] = rank

    if to_show_only_concepts:
        data = [d for d in data if episodes[d["image-file"]]]

    data = data[: num_pos[query_concept]]

    st.markdown("## K = {}".format(k))

    for group in partition_all(num_columns, data):
        cols = st.columns(num_columns)
        for i, datum in enumerate(group):
            image_file = datum["image-file"]
            image_path = IMAGE_COCO_DIR / image_file
            caption = ",".join(episodes[image_file]) or "–"
            # is_correct_str = "✓" if datum["is-query-in-image"] else "✗"
            cols[i].markdown("{} · {}".format(datum["rank"], caption))
            cols[i].image(str(image_path))
