# TODO
# - [x] load audio model
# - [x] load image model
# - [x] load attention model
# - [x] download audio dataset (spoken-coco)
# - [x] download image dataset (coco14)
# - [ ] write forward function
# - [ ] audio loader and preprocessor
# - [ ] image loader and preprocess
# - [ ] ask Leanne: last or best checkpoint?
# - [ ] check my scores agree with Leanne's
#

import pdb
import pickle
import sys

from collections import OrderedDict
from pathlib import Path

import streamlit as st
import torch

from torch import nn
from torchvision.models import alexnet

sys.path.insert(0, ".")
from models.multimodalModels import mutlimodal as AudioModel
from models.GeneralModels import ScoringAttentionModule

# from test_DH_few_shot_test_with_sampled_queries import (
#     LoadAudio as load_audio,
#     LoadImage as load_image,
# )


class MattNet:
    def __init__(self):
        folder = Path(
            "model_metadata/spokencoco_train/AudioModel-Transformer_ImageModel-Resnet50_ArgumentsHash-2560499dfc_ConfigFile-params"
        )

        with open(folder / "args.pkl", "rb") as f:
            args = pickle.load(f)

        audio_model = AudioModel(args)
        image_model = self.get_image_model(args)
        attention_model = ScoringAttentionModule(args)

        path_checkpoint = folder / "models" / "best_ckpt.pt"
        state = torch.load(path_checkpoint)

        audio_model.load_state_dict(self.fix_ddp_module(state["audio_model"]))
        image_model.load_state_dict(self.fix_ddp_module(state["image_model"]))
        attention_model.load_state_dict(self.fix_ddp_module(state["attention"]))

        self.audio_model = audio_model
        self.image_model = image_model
        self.attention_model = attention_model

    @staticmethod
    def get_image_model(args):
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


st.image("/mnt/private-share/speechDatabases/coco/val2014/COCO_val2014_000000348730.jpg")
# matt_net = MattNet()
