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
# - [ ] precompute image embeddings
# - [ ] ask Leanne: last or best checkpoint?
# - [ ] check my scores agree with Leanne's
#

import json
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
from torch import nn
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


# img_ids = coco.getImgIds(catIds=cat_ids)
# img = coco.loadImgs(img_ids[0])[0]
# img_path = coco_path / "val2014" / img["file_name"]
# st.image(str(img_path))
# for ann in anns:
#     st.image(255 * coco.annToMask(ann))
# pdb.set_trace()

concepts = load_concepts()
alignments = load_alignments(concepts)
captions_data = load_captions()

pad_kwargs = {
    "target_length": ARGS["audio_config"].get("target_length", 1024),
    "padval": ARGS["audio_config"].get("padval", 0),
}

IMG_SIZE = 256, 256
image_transforms = {
    "resize": transforms.Resize(IMG_SIZE),
    "to_tensor": transforms.ToTensor(),
    "image_normalize": transforms.Normalize(
        mean=ARGS["image_config"]["RGB_mean"],
        std=ARGS["image_config"]["RGB_std"],
    ),
}

print("Load model...")
mattnet = MattNet()
mattnet.eval()

with st.sidebar:
    query_concept = st.selectbox("query concept", concepts)
    episode_no = st.number_input(
        "episode no.", min_value=0, max_value=1000, format="%d", step=1
    )

episodes = np.load("data/test_episodes.npz", allow_pickle=True)["episodes"].item()
audio_query, _ = episodes[episode_no]["queries"][query_concept]
image_matching_set = list(episodes["matching_set"].keys())

coco = COCO(IMAGE_COCO_DIR / "annotations" / "instances_val2014.json")
coco_category_ids = coco.getCatIds(catNms=[query_concept])

audio_path = AUDIO_COCO_DIR / audio_query
audio_name = audio_path.stem

print("Load audio...")
audio, _ = load_audio(
    audio_path, alignments[audio_name][query_concept], ARGS["audio_config"]
)
audio, _ = pad_audio(audio, **pad_kwargs)

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

for i, image_file in enumerate(image_matching_set):
    contains_query_based_on_caption = query_concept in episodes["matching_set"][image_file]

    if not contains_query_based_on_caption:
        continue

    image_path = IMAGE_COCO_DIR / image_file
    image_name = image_path.stem

    # print("Load image...")
    image = load_image(image_path, **image_transforms)
    image = image.unsqueeze(0)

    # print("Predict...")
    with torch.no_grad():
        score, attention = mattnet(audio, image)

    # input_image = Image.open(image_path)
    # input_image = input_image.resize(IMG_SIZE)
    # input_image = np.array(input_image)

    attention_image = torch.sigmoid(attention).view(7, 7).numpy()
    attention_image = Image.fromarray(attention_image)
    attention_image = attention_image.resize(IMG_SIZE, Image.Resampling.BICUBIC)
    attention_image = np.clip(np.array(attention_image), 0, 1)

    st.markdown(f"image name: `{image_name}`")
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

    coco_image_id = [int(image_name.split("_")[-1])]
    coco_annot_ids = coco.getAnnIds(imgIds=coco_image_id, catIds=coco_category_ids)
    coco_annots = coco.loadAnns(coco_annot_ids)
    contains_query_based_on_image = len(coco_annots) > 0

    st.markdown(
        """
- score (maximum of attetion): {:.3f}
- contains query based on caption: {:s}
- contains query based on image: {:s}
- captions:
{:s}
""".format(
            score.item(),
            "✓" if contains_query_based_on_caption else "✗",
            "✓" if contains_query_based_on_image else "✗",
            captions_for_image_str,
        )
    )
    st.markdown("---")
