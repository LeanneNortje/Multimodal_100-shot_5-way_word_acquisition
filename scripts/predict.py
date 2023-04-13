import json
import os
import pdb
import pickle
import sys

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm

from torch import nn
from torchdata.datapipes.map import SequenceWrapper
from torchvision import transforms
from torchvision.models import alexnet

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


MODEL_DIR = Path("model_metadata/spokencoco_train/AudioModel-Transformer_ImageModel-Resnet50_ArgumentsHash-2560499dfc_ConfigFile-params")
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
    def __init__(self, device="cpu"):
        super().__init__()

        audio_model = AudioModel(ARGS)
        image_model = self.build_image_model(ARGS)
        attention_model = ScoringAttentionModule(ARGS)

        path_checkpoint = MODEL_DIR / "models" / "best_ckpt.pt"
        state = torch.load(path_checkpoint, map_location=device)

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
    image = load_image(image_path, **KWARGS_LOAD_IMAGE)
    return image


def load_audio_1(audio_file, get_alignment):
    audio_path = AUDIO_COCO_DIR / audio_file
    audio_name = audio_path.stem

    alignment = get_alignment(audio_name)

    audio, _ = load_audio(audio_path, alignment, ARGS["audio_config"])
    audio, _ = pad_audio(audio, **KWARGS_PAD_AUDIO)
    return audio


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


def compute_scores(mattnet, image_files, audio):
    image_emb = cache_np(
        "data/image_embeddings_matching_set.npy",
        compute_image_embeddings,
        network=mattnet.image_model,
        image_files=image_files,
    )
    image_emb = torch.tensor(image_emb)
    image_emb = image_emb.view(image_emb.size(0), image_emb.size(1), -1)
    image_emb = image_emb.transpose(1, 2)
    _, _, audio_emb = mattnet.audio_model(audio)

    with torch.no_grad():
        scores = mattnet.attention_model.one_to_many_score(image_emb, audio_emb, None)
        scores = scores[0].numpy()

    return scores


def main():

    mattnet = MattNet()
    mattnet.eval()

    concepts = load_concepts()
    alignments = load_alignments(concepts)

    episodes = np.load("data/test_episodes.npz", allow_pickle=True)
    episodes = episodes["episodes"].item()
    num_episodes = 1000

    image_matching_set = list(sorted(episodes["matching_set"].keys()))

    def compute1(episode, concept):
        get_alignment = lambda a: alignments[a][concept]
        audio_file, _ = episodes[episode]["queries"][concept]
        audio = load_audio_1(audio_file, get_alignment)
        concept_str = concept.replace(" ", "-")
        return cache_np(
            f"data/scores/{concept_str}-{episode}.npy",
            compute_scores,
            mattnet,
            image_matching_set,
            audio,
        )

    for episode in range(num_episodes):
        for concept in concepts:
            print("{:4d} · {:s}".format(int(episode), concept))
            compute1(episode, concept)


if __name__ == "__main__":
    main()
