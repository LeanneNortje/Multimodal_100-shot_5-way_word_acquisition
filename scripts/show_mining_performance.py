import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm

from predict import COCOData
from evaluate import COCOResults


dataset = COCOData()
results = {
    split: COCOResults("100-loc-v2-ret", dataset, split=split)
    for split in "val2014 train2014".split()
}

concepts = dataset.load_concepts()
NUM_CONCEPTS = len(concepts)

sns.set_context("notebook")

def get_split(image_file):
    split, _ = image_file.parts
    return split

def add_annotations(data):
    return {
        c: [
            datum + (results[get_split(datum[1])].is_query_in_image(concepts[c], datum[1]), )
            for datum in tqdm(data[c])
        ]
        for c in range(NUM_CONCEPTS)
    }

# @st.cache_data
def load_data():
    data = np.load("data/sampled_img_ranks_and_scores.npz", allow_pickle=True)
    data = data["record"].item()
    # data = {c: sorted(data[c], key=lambda t: -t[2]) for c in range(NUM_CONCEPTS)}
    return add_annotations(data)


data = load_data()
TOP_K = 15


# for c in range(5):
#     data1 = data[c]
#     concept = concepts[c]
#     st.markdown(concept)
#     for rank, filename, score, has_concept in data1[:TOP_K]:
#         has_concept_str = "✓" if has_concept else "✗"
#         st.markdown("{:d} · {:.2f} · {:s}".format(rank, score, has_concept_str))
#         st.code(filename)
#         st.image(str(dataset.image_dir / filename))
#     st.markdown("---")


df = [
    {
        "concept": concept,
        "rank": r,
        "score": datum[2],
        "precision": sum(datum[3] for datum in data[c][:r]) / r
    }
    for c, concept in enumerate(concepts)
    for r, datum in enumerate(data[c], start=1)
]

df = pd.DataFrame(df)
df

fig, axs = plt.subplots(ncols=2, figsize=(8, 3.5), sharey=True)
sns.lineplot(data=df, x="rank", y="precision", hue="concept", ax=axs[0], legend=False)
sns.lineplot(data=df, x="score", y="precision", hue="concept", ax=axs[1])
sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))
axs[1].invert_xaxis()
fig.tight_layout()
# fig.savefig("output/taslp/mining-precision-images-mattnet-k-100.pdf")
st.pyplot(fig)
