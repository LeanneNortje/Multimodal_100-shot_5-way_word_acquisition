import json
import os
import pdb
import pickle
import sys

from collections import OrderedDict
from pathlib import Path

import click
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


BASE_DIR = Path("/mnt/private-share/speechDatabases")
AUDIO_COCO_DIR = BASE_DIR / "spoken-coco"
IMAGE_COCO_DIR = BASE_DIR / "coco"


CONFIGS = {
    "5": {
        "num-shots": 5,
        "num-image-layers": 2,
    },
    "100": {
        "num-shots": 100,
        "num-image-layers": 2,
    },
    "100-loc": {
        "num-shots": 100,
        "num-image-layers": 0,
    },
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
    def __init__(self, config_name, device="cpu"):
        super().__init__()

        config = CONFIGS[config_name]

        num_shots = config["num-shots"]
        num_image_layers = config["num-image-layers"]

        self.num_shots = num_shots
        self.model_dir = Path(f"model_metadata/spokencoco_train/model-{config_name}")

        with open(self.model_dir / "args.pkl", "rb") as f:
            self.args = pickle.load(f)

        self.kwargs_pad_audio = {
            "target_length": self.args["audio_config"].get("target_length", 1024),
            "padval": self.args["audio_config"].get("padval", 0),
        }

        self.img_size = 256, 256
        self.kwargs_load_image = {
            "resize": transforms.Resize(self.img_size),
            "to_tensor": transforms.ToTensor(),
            "image_normalize": transforms.Normalize(
                mean=self.args["image_config"]["RGB_mean"],
                std=self.args["image_config"]["RGB_std"],
            ),
        }

        self.args["num_image_layers"] = num_image_layers

        audio_model = AudioModel(self.args)
        image_model = self.build_image_model(self.args)
        attention_model = ScoringAttentionModule(self.args)

        path_checkpoint = self.model_dir / "models" / "best_ckpt.pt"
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


    def load_image_1(self, image_file):
        image_path = IMAGE_COCO_DIR / image_file
        image = load_image(image_path, **self.kwargs_load_image)
        return image


    def load_audio_1(self, audio_file, get_alignment):
        audio_path = AUDIO_COCO_DIR / audio_file
        audio_name = audio_path.stem

        alignment = get_alignment(audio_name)

        audio, _ = load_audio(audio_path, alignment, self.args["audio_config"])
        audio, _ = pad_audio(audio, **self.kwargs_pad_audio)
        return audio


def compute_image_embeddings(mattnet, image_files):
    batch_size = 100

    dp = SequenceWrapper(image_files)
    dp = dp.map(mattnet.load_image_1)
    dp = dp.batch(batch_size=batch_size)

    mattnet.eval()

    with torch.no_grad():
        image_embeddings = (mattnet.image_model(torch.stack(batch)) for batch in tqdm(dp))
        image_embeddings = np.concatenate([e.numpy() for e in image_embeddings])

    return image_embeddings


def compute_scores(mattnet, image_files, audio, config_name):
    image_emb = cache_np(
        f"data/image_embeddings_matching_set_{config_name}.npy",
        compute_image_embeddings,
        mattnet=mattnet,
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


@click.command()
@click.option("-c", "--config", "config_name", required=True)
def main(config_name):

    mattnet = MattNet(config_name)
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
        audio = mattnet.load_audio_1(audio_file, get_alignment)
        concept_str = concept.replace(" ", "-")
        return cache_np(
            f"data/scores-{config_name}/{concept_str}-{episode}.npy",
            compute_scores,
            mattnet,
            image_matching_set,
            audio,
            config_name,
        )

    for episode in range(num_episodes):
        for concept in concepts:
            print("{:4d} · {:s}".format(int(episode), concept))
            compute1(episode, concept)


if __name__ == "__main__":
    main()
