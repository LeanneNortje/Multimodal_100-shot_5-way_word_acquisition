import os
import pdb

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

from predict import COCOData, MattNet, DavidHarwathNet
from evaluate import COCOResults

st.set_page_config(layout="wide")


TO_SAVE_DATA_FOR_PAPER = False
config_name = os.environ.get("CONFIG", "100-loc-v2-ret")

dataset = COCOData()
concepts = dataset.load_concepts()
episode_no = 0

results = COCOResults(config_name, dataset)

# Randomly initialised model
net_random = DavidHarwathNet("pretrained-cpc", to_load=(), image_model_pretrained=False)
net_random.eval()

# Model initialised with each encoder trained independently
net_independent = DavidHarwathNet("pretrained-cpc", to_load=("audio", ), image_model_pretrained=True)
net_independent.eval()

# Model initialsed with the encoders trained jointly, but on different background data
net_background = DavidHarwathNet("pretrained", to_load=("audio", "image"))
net_background.eval()

# Model initialised with the encoders trained jointly on the few-shot data
mattnet = MattNet(config_name)
mattnet.eval()


def scale_0_100_sigmoid(attention):
    attention = 5 * (attention / 100 - 0.5)
    return torch.sigmoid(attention).numpy()


def scale_min_max_linear(attention):
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    return attention.numpy()


def get_explanation(model, audio, image_path, scale_func):
    image = mattnet.load_image_1(image_path)
    image = image.unsqueeze(0)

    # original image
    image_rgb = Image.open(image_path)
    image_rgb = image_rgb.convert("RGB")

    # image_rgb = image_rgb.resize(IMG_SIZE)
    image_rgb = np.array(image_rgb) / 255
    h, w, _ = image_rgb.shape

    with torch.no_grad():
        _, attention = model(audio, image)

    # prepare attention map for visualization
    attention = attention.view(7, 7)
    explanation = scale_func(attention)

    explanation = Image.fromarray(explanation).resize((w, h))
    explanation = np.array(explanation)

    return show_cam_on_image(image_rgb, explanation, use_rgb=True)


def save_image(image, image_name, suffix):
    path = f"output/taslp/imgs/attention-random/{image_name}-attention-{suffix}.jpg"
    image_out = Image.fromarray(image)
    image_out.save(path)


for query_concept in concepts:
    audio_query, _ = results.dataset.episodes[episode_no]["queries"][query_concept]
    audio_path = dataset.get_audio_path(audio_query)
    audio_name = audio_path.stem

    alignment = dataset.alignments[audio_name][query_concept]
    audio = mattnet.load_audio_1(audio_path, alignment)

    data = results.load(query_concept, episode_no)
    data = sorted(data, reverse=True, key=lambda datum: datum["score"])

    TOP_K = 5
    data = data[:TOP_K]

    for datum in data:
        image_file = datum["image-file"]
        image_path = dataset.get_image_path(image_file)
        image_name = image_path.stem

        cols = st.columns(5)

        cols[0].markdown("input image")
        cols[0].image(str(image_path))

        image = get_explanation(net_random, audio, image_path, scale_min_max_linear)
        cols[1].markdown("model: random · scale: min-max linear")
        cols[1].image(image)
        save_image(image, image_name, "harwath-random")

        image = get_explanation(net_independent, audio, image_path, scale_min_max_linear)
        cols[2].markdown("model: independent · scale: min-max linear")
        cols[2].image(image)
        save_image(image, image_name, "harwath-independent")

        image = get_explanation(net_background, audio, image_path, scale_min_max_linear)
        cols[3].markdown("model: joint on background · scale: min-max linear")
        cols[3].image(image)
        save_image(image, image_name, "harwath-joint-background")

        image = get_explanation(mattnet, audio, image_path, scale_0_100_sigmoid)
        cols[4].markdown("model: joint on few-shot data · scale: 0:100 + σ")
        cols[4].image(image)
        save_image(image, image_name, "mattnet-joint-few-shot")

    st.markdown("---")