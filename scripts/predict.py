import json
import os
import pdb
import pickle
import sys

from collections import OrderedDict
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

from tqdm import tqdm

from torch import nn
from torchdata.datapipes.map import SequenceWrapper
from torchvision import transforms
from torchvision.models import alexnet

from toolz import dissoc

from utils import load

sys.path.insert(0, ".")
from models.multimodalModels import mutlimodal as AudioModel
from models.GeneralModels import ScoringAttentionModule

from test_DH_few_shot_test_with_sampled_queries import (
    LoadAudio as load_audio,
    LoadImage as load_image,
    PadFeat as pad_audio,
)


class Dataset:
    NUM_EPISODES = 1000

    @cached_property
    def episodes(self):
        episodes = np.load(self.path_episodes, allow_pickle=True)
        return episodes["episodes"].item()

    @cached_property
    def image_matching_set(self):
        # previous implementation:
        # >>> matching_set = self.episodes["matching_set"].keys()
        #
        # the following is not really correct (if we consider the retreival task),
        # since there are no background images in the matching sets corresponding to each episode
        # matching_set = [
        #     img
        #     for e in range(self.NUM_EPISODES)
        #     for img in self.episodes[e]["matching_set"].values()
        # ]
        # matching_set = set(matching_set)
        # return list(sorted(matching_set))
        return self.episodes["matching_set"].keys()


class COCOData(Dataset):
    def __init__(self):
        from test_DH_few_shot_test_with_sampled_queries import (
            load_concepts,
            load_alignments,
        )

        self.base_dir = Path("/mnt/private-share/speechDatabases")
        self.audio_dir = self.base_dir / "spoken-coco"
        self.image_dir = self.base_dir / "coco"

        self.load_concepts = load_concepts
        self.load_alignments = load_alignments

        self.path_episodes = "data/test_episodes.npz"

    @cached_property
    def alignments(self):
        return self.load_alignments()

    def load_captions(self):
        with open(self.audio_coco_dir / "SpokenCOCO_val.json", "r") as f:
            return json.load(f)["data"]

    def get_audio_path(self, audio_file):
        return self.audio_dir / audio_file

    def get_image_path(self, image_file):
        return self.image_dir / image_file

    def get_audio_path_episode_concept(self, episode, concept):
        audio_file, _ = self.episodes[episode]["queries"][concept]
        return self.get_audio_path(audio_file)

    def get_alignment_concept(self, audio_name, concept):
        return self.alignments[audio_name][concept]


class FlickrData(Dataset):
    def __init__(self):
        self.base_dir = Path("/home/doneata/data")
        self.audio_dir = self.base_dir / "flickr8k-audio" / "wavs"
        self.image_dir = self.base_dir / "flickr8k-images" / "Flicker8k_Dataset"

        self.base_metadata_dir = Path("/home/doneata/work/mattnet-yfacc")
        self.path_episodes = (
            self.base_metadata_dir
            / "low-resource_support_sets"
            / "data"
            / "test_episodes.npz"
        )

    @cached_property
    def alignments(self):
        return self.load_alignments()

    @staticmethod
    def reformat_key(key):
        # from `271637337_0700f307cf.jpg#2` to `271637337_0700f307cf_2`
        # TODO: probably could use a tuple or namedtuple to hold a key
        key, num = key.split("#")
        key = key.split(".")[0] + "_" + num
        return key

    @staticmethod
    def parse_ctm(line):
        key, _, time_start, duration, word = line.split()
        key = FlickrData.reformat_key(key)
        time_start = int(100 * float(time_start))
        duration = int(100 * float(duration))
        return {
            "key": key,
            "time-start": time_start,
            "time-end": time_start + duration,
            "word": word.lower(),
        }

    @staticmethod
    def parse_token(line):
        key, *words = line.split()
        key = FlickrData.reformat_key(key)
        text = " ".join(words)
        return (key, text)

    @staticmethod
    def load_alignments():
        path = "/home/doneata/work/herman-semantic-flickr/data/flickr_8k.ctm"
        alignments_list = load(path, FlickrData.parse_ctm)
        alignments_dict = {
            key: [dissoc(d, "key") for d in group]
            for key, group in groupby(alignments_list, key=lambda x: x["key"])
        }
        return alignments_dict

    def load_concepts(self):
        path = (
            self.base_metadata_dir
            / "low-resource_support_sets"
            / "data"
            / "test_keywords.txt"
        )
        return load(path, lambda line: line.strip())

    @cached_property
    def captions(self):
        path = self.base_dir / "flickr8k-text" / f"Flickr8k.token.txt"
        return dict(load(path, self.parse_token))

    def get_audio_path(self, audio_file):
        return self.audio_dir / (audio_file + ".wav")

    def get_image_path(self, image_file):
        return self.image_dir / (image_file + ".jpg")

    def get_audio_path_episode_concept(self, episode, concept):
        audio_file = self.episodes[episode]["queries"][concept]
        return self.get_audio_path(audio_file)

    def get_alignment_concept(self, audio_name, concept):
        for a in self.alignments[audio_name]:
            if a["word"] == concept:
                return a["time-start"], a["time-end"]
        raise ValueError


class YFACCData(Dataset):
    pass


CONFIGS = {
    "5": {
        "num-shots": 5,
        "num-image-layers": 2,
        "data-class": COCOData,
        "task": "retrieval",
        "model-name": "5",
    },
    "100": {
        "num-shots": 100,
        "num-image-layers": 2,
        "data-class": COCOData,
        "task": "retrieval",
        "model-name": "100",
    },
    "100-loc": {
        "num-shots": 100,
        "num-image-layers": 0,
        "data-class": COCOData,
        "task": "retrieval",
        "model-name": "100-loc",
    },
    "flickr-en-5-cls": {
        "num-shots": 5,
        "num-image-layers": 0,
        "data-class": FlickrData,
        "task": "classification",
        "model-name": "flickr-en-5",
    },
    "flickr-yo-5-cls": {
        "num-shots": 5,
        "num-image-layers": 0,
        "data-class": YFACCData,
        "task": "classification",
        "model-name": "flickr-yo-5",
    },
}


def cache_np(path, func, *args, **kwargs):
    if os.path.exists(path):
        return np.load(path)
    else:
        result = func(*args, **kwargs)
        np.save(path, result)
        return result


class MattNet(nn.Module):
    def __init__(self, config_name, device="cpu"):
        super().__init__()

        config = CONFIGS[config_name]
        model_name = config["model-name"]

        num_shots = config["num-shots"]
        num_image_layers = config["num-image-layers"]

        self.num_shots = num_shots
        self.model_dir = Path(f"model_metadata/spokencoco_train/model-{model_name}")

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

    def load_image_1(self, image_path):
        image = load_image(image_path, **self.kwargs_load_image)
        return image

    def load_audio_1(self, audio_path, alignment):
        audio, _ = load_audio(audio_path, alignment, self.args["audio_config"])
        audio, _ = pad_audio(audio, **self.kwargs_pad_audio)
        return audio


def compute_image_embeddings(mattnet, image_paths):
    batch_size = 100

    dp = SequenceWrapper(image_paths)
    dp = dp.map(mattnet.load_image_1)
    dp = dp.batch(batch_size=batch_size)

    mattnet.eval()

    with torch.no_grad():
        image_embeddings = (
            mattnet.image_model(torch.stack(batch)) for batch in tqdm(dp)
        )
        image_embeddings = np.concatenate([e.numpy() for e in image_embeddings])

    return image_embeddings


def compute_scores(mattnet, image_paths, audio, config_name):
    image_emb = cache_np(
        f"data/image_embeddings_matching_set_{config_name}.npy",
        compute_image_embeddings,
        mattnet=mattnet,
        image_paths=image_paths,
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
@click.option(
    "-c", "--config", "config_name", required=True, type=click.Choice(CONFIGS)
)
def main(config_name):
    config = CONFIGS[config_name]
    dataset = config["data-class"]()

    task = config["task"]

    mattnet = MattNet(config_name)
    mattnet.eval()

    concepts = dataset.load_concepts()

    if task == "retrieval":
        image_paths = [
            dataset.get_image_path(image) for image in dataset.image_matching_set
        ]

        def get_image_paths(episode):
            return image_paths

    elif task == "classification":

        def get_image_paths(episode):
            return [
                dataset.get_image_path(im)
                for im in dataset.episodes[episode]["matching_set"].values()
            ]

    else:
        raise ValueError(f"Unknown task: {task}")

    def compute1(episode, concept):
        concept_str = concept.replace(" ", "-")

        audio_path = dataset.get_audio_path_episode_concept(episode, concept)
        alignment = dataset.get_alignment_concept(audio_path.stem, concept)
        audio = mattnet.load_audio_1(audio_path, alignment)
        image_paths = get_image_paths(episode)

        return cache_np(
            f"data/scores-{config_name}/{concept_str}-{episode}.npy",
            compute_scores,
            mattnet,
            image_paths,
            audio,
            config_name,
        )

    for episode in range(dataset.NUM_EPISODES):
        for concept in concepts:
            print("{:4d} · {:s}".format(int(episode), concept))
            compute1(episode, concept)


if __name__ == "__main__":
    main()
