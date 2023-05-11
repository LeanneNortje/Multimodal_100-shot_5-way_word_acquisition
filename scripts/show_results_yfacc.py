import json
import os
import pdb

from itertools import groupby
from pathlib import Path

from toolz import dissoc, second, partition_all
from matplotlib import pyplot as plt

import librosa
import numpy as np
import streamlit as st

import textgrid

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
    def __init__(self, key, *, split, text_en, text_yo, alignments_en):
        self.key = key
        # fmt: off
        self.path_image = BASE_DIR_EN / "flickr8k-images" / (self.get_image_key(key) + ".jpg")
        self.path_audio_en = BASE_DIR_EN / "flickr8k-audio" / "wavs" / (self.key + ".wav")
        self.path_audio_yo = BASE_DIR_YO / f"flickr_audio_yoruba_{split}" / ("S001_" + self.key + ".wav")
        # fmt: on
        self.text_en = text_en
        self.text_yo = text_yo
        self.alignments_en = alignments_en
        self.alignments_yo = self.load_alignments_yo(key)

    def get_image_key(self, key):
        parts = key.split("_")
        return "_".join(parts[:2])

    def load_alignments_yo(self, key):
        path = BASE_DIR_YO / "Flickr8k_alignment" / (key + ".TextGrid")
        if os.path.exists(path):
            return [
                {
                    "time-start": int(1000 * i.minTime),
                    "time-end": int(1000 * i.maxTime),
                    "word": i.mark.casefold(),
                }
                for i in textgrid.TextGrid.fromFile(path)[0]
                if i.mark
            ]
        else:
            return []


class FlickrSamples:
    def __init__(self, split):
        assert split in "train dev test".split()
        self.split = split

        texts_en = self.load_texts_en(split)
        texts_yo = self.load_texts_yo(split)
        alignments_en = self.load_alignments_en()

        keys = texts_yo.keys()

        self.samples = [
            Sample(
                key,
                split=split,
                text_en=texts_en[key],
                text_yo=texts_yo[key],
                alignments_en=alignments_en.get(key, []),
            )
            for key in keys
        ]

    @staticmethod
    def reformat_key(key):
        # from `271637337_0700f307cf.jpg#2` to `271637337_0700f307cf_2`
        # TODO: probably could use a tuple or namedtuple to hold a key
        key, num = key.split("#")
        key = key.split(".")[0] + "_" + num
        return key

    @staticmethod
    def parse_token(line):
        key, *words = line.split()
        key = FlickrSamples.reformat_key(key)
        text = " ".join(words)
        return (key, text)

    @staticmethod
    def parse_ctm(line):
        key, _, time_start, duration, word = line.split()
        key = FlickrSamples.reformat_key(key)
        time_start = int(1000 * float(time_start))
        duration = int(1000 * float(duration))
        return {
            "key": key,
            "time-start": time_start,
            "time-end": time_start + duration,
            "word": word.lower(),
        }

    def load_texts_en(self, split):
        path = BASE_DIR_EN / "flickr8k-text" / f"Flickr8k.token.txt"
        return dict(load(path, self.parse_token))

    def load_texts_yo(self, split):
        path = BASE_DIR_YO / "Flickr8k_text" / f"Flickr8k.token.{split}_yoruba.txt"
        return dict(load(path, self.parse_token))

    def load_alignments_en(self):
        path = "/home/doneata/work/herman-semantic-flickr/data/flickr_8k.ctm"
        alignments_list = load(path, self.parse_ctm)
        alignments_dict = {
            key: [dissoc(d, "key") for d in group]
            for key, group in groupby(alignments_list, key=lambda x: x["key"])
        }
        return alignments_dict

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
        st.markdown(f"## key: `{sample.key}`")
        st.image(str(sample.path_image))
        st.audio(str(sample.path_audio_en))
        st.code(sample.text_en)
        st.code(json.dumps(sample.alignments_en, indent=4))
        st.audio(str(sample.path_audio_yo))
        st.code(sample.text_yo)
        st.code(json.dumps(sample.alignments_yo, indent=4, ensure_ascii=False))
        st.markdown("---")


def main():
    show_samples()


if __name__ == "__main__":
    main()
