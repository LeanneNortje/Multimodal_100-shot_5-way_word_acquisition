# Example run:
# streamlit run scripts/show_image_localisation.py

# TODO
# - [x] load audio model
# - [x] load image model
# - [x] load attention model
# - [x] download audio dataset (spoken-coco)
# - [x] download image dataset (coco14)
# - [x] audio loader and preprocessor
# - [x] image loader and preprocessor
# - [x] load word alignments
# - [x] write forward function
# - [x] precompute image embeddings
# - [x] check that my scores match Leanne's
# - [?] overlay attention on top of images
# - [x] add caching in streamlit (see st.cache_data)
# - [ ] run on GPU :-)
# - [ ] ask Leanne: last or best checkpoint?
#

import json
import os
import pdb
import pickle
import sys

from collections import OrderedDict
from pathlib import Path
from PIL import Image

import numpy as np
import streamlit as st
import torch

from toolz import first
from tqdm import tqdm

from torch import nn
from torchdata.datapipes.map import SequenceWrapper
from torchvision import transforms
from torchvision.models import alexnet

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

from predict import COCOData, MattNet
from evaluate import COCOResults

st.set_page_config(layout="wide")


TO_SAVE_DATA_FOR_PAPER = False
config_name = os.environ.get("CONFIG", "100-loc-v2-ret")

dataset = COCOData()
concepts = dataset.load_concepts()

results = COCOResults(config_name, dataset)
mattnet = MattNet(config_name)
mattnet.eval()


with st.sidebar:
    query_concept = st.selectbox("query concept", concepts)
    episode_no = st.number_input(
        "episode no.", min_value=0, max_value=1000, format="%d", step=1
    )
    vis_type = st.selectbox("explanation", ["attention", "grad-cam"])
    τ = st.slider("threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

audio_query, _ = results.dataset.episodes[episode_no]["queries"][query_concept]
audio_path = dataset.get_audio_path(audio_query)
audio_name = audio_path.stem

alignment = dataset.alignments[audio_name][query_concept]
audio = mattnet.load_audio_1(audio_path, alignment)

if TO_SAVE_DATA_FOR_PAPER:
    import librosa
    import seaborn as sns

    from matplotlib import pyplot as plt

    sample_rate = 16_000
    window_stride = 0.01

    y, sr = librosa.load(audio_path, sr=sample_rate)

    k = window_stride * sample_rate
    α, ω = alignment
    α = int(k * α)
    ω = int(k * ω)
    y = y[α:ω]

    fig, ax = plt.subplots()
    ax.plot(y)
    ax.axis("off")

    st.code(audio_path)
    st.audio(y, sample_rate=sr)
    st.pyplot(fig)

    concept_str = query_concept.replace(" ", "-")
    fig.savefig(f"output/taslp/imgs/{config_name}/audio-{concept_str}.png")

st.markdown(f"query concept: `{query_concept}`")
st.markdown(f"audio name: `{audio_name}`")
st.audio(str(audio_path))
st.markdown("caption:")
st.code(dataset.captions[audio_query])
st.markdown("---")

data = results.load(query_concept, episode_no)
data = sorted(data, reverse=True, key=lambda datum: datum["score"])

for rank, datum in enumerate(data, start=1):
    datum["rank"] = rank

TOP_K = 5
# data = [
#     datum
#     for datum in data
#     if datum["contains-query-based-on-image"]
#     and not datum["contains-query-based-on-caption"]
# ]
data = data[:TOP_K]


class MattNetForGradCAM(nn.Module):
    def __init__(self, mattnet, audio):
        super().__init__()
        self.mattnet = mattnet
        self.audio = audio

    def forward(self, image):
        score, _ = self.mattnet(self.audio, image)
        return [score]


mattnet_for_gradcam = MattNetForGradCAM(mattnet, audio)
grad_cam = GradCAM(
    model=mattnet_for_gradcam,
    target_layers=[mattnet_for_gradcam.mattnet.image_model[-1]],
)
targets = [RawScoresOutputTarget()]

for datum in data:
    image_file = datum["image-file"]
    image_path = dataset.get_image_path(image_file)
    image_name = image_path.stem

    image = mattnet.load_image_1(image_path)
    image = image.unsqueeze(0)

    with torch.no_grad():
        score, attention = mattnet(audio, image)

    # original image
    image_rgb = Image.open(image_path)
    image_rgb = image_rgb.convert("RGB")

    # image_rgb = image_rgb.resize(IMG_SIZE)
    image_rgb = np.array(image_rgb) / 255
    h, w, _ = image_rgb.shape

    # prepare attention map for visualization
    if vis_type == "attention":
        attention = attention.view(7, 7)
        attention = 5 * (attention / 100 - 0.5)
        explanation = torch.sigmoid(attention).numpy()
    elif vis_type == "grad-cam":
        explanation = grad_cam(input_tensor=image, targets=targets)[0]
    else:
        assert False

    explanation = Image.fromarray(explanation).resize((w, h))
    explanation = np.array(explanation)
    explanation_binary = explanation > τ
    explanation_binary = 255 * explanation_binary.astype(np.uint8)
    image_explanation = show_cam_on_image(image_rgb, explanation, use_rgb=True)

    if TO_SAVE_DATA_FOR_PAPER:
        import shutil

        shutil.copy(image_path, f"output/taslp/imgs/{config_name}/{image_name}.jpg")
        image_explanation_out = Image.fromarray(image_explanation)
        image_explanation_out.save(
            f"output/taslp/imgs/{config_name}/{image_name}-explanation-{vis_type}.jpg"
        )

    # annotations
    coco_annots = results.get_coco_annots(image_file, query_concept)
    if len(coco_annots) > 0:
        masks = [results.coco.annToMask(a) for a in coco_annots]
        masks = np.stack(masks)
        image_annots = 255 * (masks.sum(axis=0) > 0)
    else:
        image_annots = np.zeros(image_rgb.shape)

    st.markdown("rank: {}".format(datum["rank"]))
    st.markdown("image name: `{}`".format(image_name))

    cols = st.columns(4)

    cols[0].markdown("image")
    cols[0].image(str(image_path))

    cols[1].markdown("explanation: " + vis_type)
    cols[1].image(image_explanation)

    cols[2].markdown("explanation (binary)")
    cols[2].image(explanation_binary)

    cols[3].markdown("annotations")
    cols[3].image(image_annots)

    captions_for_image = dataset.captions_image[image_file]
    captions_for_image_str = "\n".join(f"  - `{c}`" for c in captions_for_image)

    st.markdown(
        """
- score (maximum of attention): {:.3f}
- contains query based on caption: {:s}
- contains query based on image: {:s}
- captions:
{:s}
""".format(
            score.item(),
            "✓" if datum["is-query-in-caption"] else "✗",
            "✓" if datum["is-query-in-image"] else "✗",
            captions_for_image_str,
        )
    )
    st.markdown("---")
