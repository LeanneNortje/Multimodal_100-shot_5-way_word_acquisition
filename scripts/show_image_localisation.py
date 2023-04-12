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
# - [ ] overlay attention on top of images
# - [ ] ask Leanne: last or best checkpoint?
# - [ ] check that my scores match Leanne's
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

from pycocotools.coco import COCO

sys.path.insert(0, ".")
from models.multimodalModels import mutlimodal as AudioModel
from models.GeneralModels import ScoringAttentionModule

from test_DH_few_shot_test_with_sampled_queries import (
    load_concepts,
    load_alignments,
    LoadAudio as load_audio,
    LoadImage as load_image,
    PadFeat as pad_audio,
)


MODEL_DIR = Path(
    "model_metadata/spokencoco_train/AudioModel-Transformer_ImageModel-Resnet50_ArgumentsHash-2560499dfc_ConfigFile-params"
)


BASE_DIR = Path("/mnt/private-share/speechDatabases")
AUDIO_COCO_DIR = BASE_DIR / "spoken-coco"
IMAGE_COCO_DIR = BASE_DIR / "coco"


with open(MODEL_DIR / "args.pkl", "rb") as f:
    ARGS = pickle.load(f)


KWARGS_PAD_AUDIO = {
    "target_length": ARGS["audio_config"].get("target_length", 1024),
    "padval": ARGS["audio_config"].get("padval", 0),
}

IMG_SIZE = 256, 256
KWARGS_LOAD_IMAGE = {
    "resize": transforms.Resize(IMG_SIZE),
    "to_tensor": transforms.ToTensor(),
    "image_normalize": transforms.Normalize(
        mean=ARGS["image_config"]["RGB_mean"],
        std=ARGS["image_config"]["RGB_std"],
    ),
}


def cache_np(path, func, *args, **kwargs):
    if os.path.exists(path):
        return np.load(path)
    else:
        result = func(*args, **kwargs)
        np.save(path, result)
        return result


def load_captions():
    with open(AUDIO_COCO_DIR / "SpokenCOCO_val.json", "r") as f:
        return json.load(f)["data"]


class MattNet(nn.Module):
    def __init__(self):
        super().__init__()

        audio_model = AudioModel(ARGS)
        image_model = self.build_image_model(ARGS)
        attention_model = ScoringAttentionModule(ARGS)

        path_checkpoint = MODEL_DIR / "models" / "best_ckpt.pt"
        state = torch.load(path_checkpoint)

        audio_model.load_state_dict(self.fix_ddp_module(state["audio_model"]))
        image_model.load_state_dict(self.fix_ddp_module(state["image_model"]))
        attention_model.load_state_dict(self.fix_ddp_module(state["attention"]))

        self.audio_model = audio_model
        self.image_model = image_model
        self.attention_model = attention_model

    @staticmethod
    def build_image_model(args):
        seed_model = alexnet(pretrained=True)
        image_model = nn.Sequential(*list(seed_model.features.children()))

        last_layer_index = len(list(image_model.children()))
        image_model.add_module(
            str(last_layer_index),
            nn.Conv2d(
                256,
                args["audio_model"]["embedding_dim"],
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
        )
        return image_model

    @staticmethod
    def fix_ddp_module(state):
        # remove 'module.' of DistributedDataParallel (DDP)
        def rm_prefix(key):
            SEP = "."
            prefix, *rest = key.split(SEP)
            assert prefix == "module"
            return SEP.join(rest)

        return OrderedDict([(rm_prefix(k), v) for k, v in state.items()])

    def forward(self, audio, image):
        image_emb = self.image_model(image)
        image_emb = image_emb.view(image_emb.size(0), image_emb.size(1), -1)
        image_emb = image_emb.transpose(1, 2)
        _, _, audio_emb = self.audio_model(audio)
        att = self.attention_model.get_attention(image_emb, audio_emb, None)
        score = att.max()
        return score, att


def load_image_1(image_file):
    image_path = IMAGE_COCO_DIR / image_file
    image_name = image_path.stem
    image = load_image(image_path, **KWARGS_LOAD_IMAGE)
    return image


def compute_image_embeddings(network, image_files):
    batch_size = 100

    dp = SequenceWrapper(image_files)
    dp = dp.map(load_image_1)
    dp = dp.batch(batch_size=batch_size)

    network.eval()

    with torch.no_grad():
        image_embeddings = (network(torch.stack(batch)) for batch in tqdm(dp))
        image_embeddings = np.concatenate([e.numpy() for e in image_embeddings])

    return image_embeddings


def compute_scores(network, image_files, audio):
    image_emb = cache_np(
        "data/image_embeddings_matching_set.npy",
        compute_image_embeddings,
        network=mattnet.image_model,
        image_files=image_matching_set,
    )
    image_emb = torch.tensor(image_emb)
    image_emb = image_emb.view(image_emb.size(0), image_emb.size(1), -1)
    image_emb = image_emb.transpose(1, 2)
    _, _, audio_emb = mattnet.audio_model(audio)

    with torch.no_grad():
        scores = mattnet.attention_model.one_to_many_score(image_emb, audio_emb, None)
        scores = scores[0].numpy()

    return scores


concepts = load_concepts()
alignments = load_alignments(concepts)
captions_data = load_captions()

with st.sidebar:
    query_concept = st.selectbox("query concept", concepts)
    episode_no = st.number_input(
        "episode no.", min_value=0, max_value=1000, format="%d", step=1
    )

episodes = np.load("data/test_episodes.npz", allow_pickle=True)["episodes"].item()
audio_query, _ = episodes[episode_no]["queries"][query_concept]
image_matching_set = list(sorted(episodes["matching_set"].keys()))

coco = COCO(IMAGE_COCO_DIR / "annotations" / "instances_val2014.json")
coco_category_ids = coco.getCatIds(catNms=[query_concept])

audio_path = AUDIO_COCO_DIR / audio_query
audio_name = audio_path.stem

audio, _ = load_audio(
    audio_path, alignments[audio_name][query_concept], ARGS["audio_config"]
)
audio, _ = pad_audio(audio, **KWARGS_PAD_AUDIO)

print("Load model...")
mattnet = MattNet()
mattnet.eval()


@st.cache_data
def load_results(query_concept, episode_no):
    scores = cache_np(
        f"data/scores-{query_concept}-{episode_no}.npy",
        compute_scores,
        mattnet,
        image_matching_set,
        audio,
    )

    def contains_query_based_on_caption(image_file):
        return query_concept in episodes["matching_set"][image_file]

    def contains_query_based_on_image(image_file):
        image_name = Path(image_file).stem
        coco_image_id = [int(image_name.split("_")[-1])]
        coco_annot_ids = coco.getAnnIds(imgIds=coco_image_id, catIds=coco_category_ids)
        coco_annots = coco.loadAnns(coco_annot_ids)
        return len(coco_annots) > 0

    data = [
        {
            "score": scores[i],
            "image-file": image_file,
            "contains-query-based-on-caption": contains_query_based_on_caption(image_file),
            "contains-query-based-on-image": contains_query_based_on_image(image_file),
        }
        for i, image_file in enumerate(image_matching_set)
    ]

    data = sorted(data, reverse=True, key=lambda datum: datum["score"])

    for rank, datum in enumerate(data, start=1):
        datum["rank"] = rank

    return data


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

TOP_K = 64
data = load_results(query_concept, episode_no)
data = [datum for datum in data if datum["contains-query-based-on-image"] and not datum["contains-query-based-on-caption"]]
data = data[:TOP_K]

for datum in data:

    image_file = datum["image-file"]
    image_path = IMAGE_COCO_DIR / image_file
    image_name = image_path.stem

    image = load_image_1(image_file)
    image = image.unsqueeze(0)

    with torch.no_grad():
        score, attention = mattnet(audio, image)

    # input_image = Image.open(image_path)
    # input_image = input_image.resize(IMG_SIZE)
    # input_image = np.array(input_image)

    attention_image = torch.sigmoid(attention).view(7, 7).numpy()
    attention_image = Image.fromarray(attention_image)
    attention_image = attention_image.resize(IMG_SIZE, Image.Resampling.BICUBIC)
    attention_image = np.clip(np.array(attention_image), 0, 1)

    st.markdown("rank: {}".format(datum["rank"]))
    st.markdown("image name: `{}`".format(image_name))
    cols = st.columns(2)

    cols[0].markdown("image")
    cols[0].image(str(image_path))

    cols[1].markdown("attention")
    cols[1].image(attention_image)

    captions_for_image = [
        caption["text"]
        for entry in captions_data
        for caption in entry["captions"]
        if entry["image"] == image_file
    ]
    captions_for_image_str = "\n".join(f"  - `{c}`" for c in captions_for_image)

    # visualize annotations
    # for ann in coco_annots:
    #     st.image(255 * coco.annToMask(ann))

    st.markdown(
        """
- score (maximum of attetion): {:.3f}
- contains query based on caption: {:s}
- contains query based on image: {:s}
- captions:
{:s}
""".format(
            score.item(),
            "✓" if datum["contains-query-based-on-caption"] else "✗",
            "✓" if datum["contains-query-based-on-image"] else "✗",
            captions_for_image_str,
        )
    )
    st.markdown("---")
