import os
import pdb

from pathlib import Path

from toolz import second, partition_all
from matplotlib import pyplot as plt

import librosa
import numpy as np
import streamlit as st

# import textgrid

# st.set_page_config(layout="wide")


BASE_DIR = Path("/home/doneata/data")
BASE_DIR_EN = BASE_DIR
BASE_DIR_YO = BASE_DIR / "flickr8k-yoruba" / "Flickr8k_Yoruba_v6"


def load_vocab():
    path = BASE_DIR_YO / "Flickr8k_text" / "keywords.8_yoruba.txt"

    def parse(line):
        word_en, *words_yo = line.split()
        return word_en, " ".join(words_yo)

    return load(path, parse)


def load(path, parser):
    with open(path, "r") as f:
        return list(map(parser, f.readlines()))


class Sample:
    def __init__(self, key, *, split, text_en, text_yo):
        self.key = key
        self.path_image = (
            BASE_DIR_EN / "flickr8k-images" / (self.get_image_key(key) + ".jpg")
        )
        self.path_audio_en = (
            BASE_DIR_EN / "flickr8k-audio" / "wavs" / (self.key + ".wav")
        )
        self.path_audio_yo = (
            BASE_DIR_YO
            / f"flickr_audio_yoruba_{split}"
            / ("S001_" + self.key + ".wav")
        )
        self.text_en = text_en
        self.text_yo = text_yo

    def get_image_key(self, key):
        parts = key.split("_")
        return "_".join(parts[:2])


class FlickrSamples:
    def __init__(self, split):
        assert split in "train dev test".split()
        self.split = split
        self.texts_en = self.load_texts_en(split)
        self.texts_yo = self.load_texts_yo(split)
        keys = self.texts_yo.keys()
        self.samples = [
            Sample(
                key,
                split=split,
                text_en=self.texts_en[key],
                text_yo=self.texts_yo[key],
            )
            for key in keys
        ]

    @staticmethod
    def parse_token(line):
        key, *words = line.split()
        key, num = key.split("#")
        key = key.split(".")[0] + "_" + num
        num = int(num)
        text = " ".join(words)
        return (key, text)

    def load_texts_en(self, split):
        path = BASE_DIR_EN / "flickr8k-text" / f"Flickr8k.token.txt"
        return dict(load(path, self.parse_token))

    def load_texts_yo(self, split):
        path = BASE_DIR_YO / "Flickr8k_text" / f"Flickr8k.token.{split}_yoruba.txt"
        return dict(load(path, self.parse_token))

    # def load_alignment(key):
    #     path = BASE_DIR_YO / "Flickr8k_alignment" / key + ".TextGrid"
    #     return [
    #         ((int(i.minTime * 100), int(i.maxTime * 100)), i.mark.casefold())
    #         for i in textgrid.TextGrid.fromFile(path)[0]
    #     ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def load_predictions(model):
    pass


vocab = load_vocab()
num_words = len(vocab)

id_to_word_en = {i: w for i, (w, _) in enumerate(vocab)}
id_to_word_yo = {i: w for i, (_, w) in enumerate(vocab)}


def show_samples():
    samples = FlickrSamples("test")
    for i in range(10):
        sample = samples[i]
        st.markdown(f"## {sample.key}")
        st.image(str(sample.path_image))
        st.audio(str(sample.path_audio_en))
        st.code(sample.text_en)
        st.audio(str(sample.path_audio_yo))
        st.code(sample.text_yo)
        st.markdown("---")


def main():
    show_samples()


if __name__ == "__main__":
    main()
