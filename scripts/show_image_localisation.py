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

from predict import (
    AUDIO_COCO_DIR,
    IMAGE_COCO_DIR,
    MattNet,
    load_alignments,
    load_audio_1,
    load_captions,
    load_concepts,
    load_image_1,
)
from evaluate import Results

st.set_page_config(layout="wide")


@st.cache_data
def load_resources():
    concepts = load_concepts()
    return (
        concepts,
        load_alignments(concepts),
        load_captions(),
        Results(),
        MattNet(),
    )


concepts, alignments, captions_data, results, mattnet = load_resources()
mattnet.eval()


with st.sidebar:
    query_concept = st.selectbox("query concept", concepts)
    episode_no = st.number_input(
        "episode no.", min_value=0, max_value=1000, format="%d", step=1
    )

audio_query, _ = results.episodes[episode_no]["queries"][query_concept]
audio_path = AUDIO_COCO_DIR / audio_query
audio_name = audio_path.stem

get_alignments = lambda a: alignments[a][query_concept]
audio = load_audio_1(audio_query, get_alignments)

caption = first(
    [
        caption["text"]
        for entry in captions_data
        for caption in entry["captions"]
        if caption["wav"] == audio_query
    ]
)

st.markdown(f"query concept: `{query_concept}`")
st.markdown(f"audio name: `{audio_name}`")
st.audio(str(audio_path))
st.markdown("caption:")
st.code(caption)
st.markdown("---")

data = results.load(query_concept, episode_no)
data = sorted(data, reverse=True, key=lambda datum: datum["score"])

for rank, datum in enumerate(data, start=1):
    datum["rank"] = rank

TOP_K = 64
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
    image_path = IMAGE_COCO_DIR / image_file
    image_name = image_path.stem

    image = load_image_1(image_file)
    image = image.unsqueeze(0)

    with torch.no_grad():
        score, attention = mattnet(audio, image)

    # original image
    image_rgb = Image.open(image_path)
    # image_rgb = image_rgb.resize(IMG_SIZE)
    image_rgb = np.array(image_rgb) / 255

    # prepare attention map for visualization
    # attention_image = torch.sigmoid(attention).view(8, 7).numpy()
    # attention_image = Image.fromarray(attention_image)
    # attention_image = attention_image.resize(IMG_SIZE, Image.Resampling.BICUBIC)
    # attention_image = np.clip(np.array(attention_image), 0, 1)

    # explanations
    h, w, _ = image_rgb.shape
    explanation = grad_cam(input_tensor=image, targets=targets)[0]
    explanation = Image.fromarray(explanation).resize((w, h))
    explanation = np.array(explanation)
    image_explanation = show_cam_on_image(image_rgb, explanation, use_rgb=True)

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

    cols = st.columns(3)

    cols[0].markdown("image")
    cols[0].image(str(image_path))

    cols[1].markdown("explanation")
    cols[1].image(image_explanation)

    cols[2].markdown("annotations")
    cols[2].image(image_annots)

    captions_for_image = [
        caption["text"]
        for entry in captions_data
        for caption in entry["captions"]
        if entry["image"] == image_file
    ]
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
