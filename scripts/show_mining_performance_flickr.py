from typing import List, TypedDict

import pdb
import re

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from collections import Counter
from itertools import groupby
from functools import partial
from toolz import assoc

from predict import FlickrEnData, FlickrYoData
from utils import load

sns.set_context("poster")

Datum = TypedDict(
    "Datum",
    {
        "concept": str,
        "rank": int,
        "has-concept": bool,
        "filename": str,
        "language": str,
    },
)


datasets = {
    "english": FlickrEnData(),
    "yoruba": FlickrYoData(),
}

LANGUAGES = sorted(datasets.keys())


def load_data() -> List[Datum]:
    def parse_mining_file(lang, line):
        line = line.strip()
        parts = line.split(" ")
        if len(parts) == 1:
            filename = parts[0]
            start = None
            end = None
        if len(parts) == 3:
            filename, start, end = parts
            start = float(start) / 16_000 * 100
            end = float(end) / 16_000 * 100
        if lang == "yoruba":
            fst, *rest = filename.split("_")
            assert fst == "S001"
            filename = "_".join(rest)
        return filename, start, end

    def load1(concept, lang):
        folder = datasets["english"].base_metadata_dir
        path = (
            folder
            / ("5-shot_5-way_" + lang)
            / "data"
            / "audio_pair_lists"
            / (concept + ".txt")
        )
        return load(path, partial(parse_mining_file, lang))

    def has_concept(lang, concept, filename):
        # if lang == "yoruba" and concept == "bike":
        #     pdb.set_trace()
        caption = datasets[lang].captions[filename]
        sentence = caption.lower()
        if lang == "yoruba":
            concept = datasets[lang].concept_to_yoruba[concept]
        regex_pattern = "([\s,.:;']|^)" + concept + "([\s,.:;']|$)"
        return bool(re.search(regex_pattern, sentence))

    concepts = datasets["english"].load_concepts()
    languages = datasets.keys()

    return [
        {
            "concept": concept,
            "rank": rank,
            "has-concept": has_concept(lang, concept, filename),
            "filename": filename,
            "language": lang,
            "caption": datasets[lang].captions[filename],
            "time-start": start,
            "time-end": end,
        }
        for concept in concepts
        for lang in languages
        for rank, (filename, start, end) in enumerate(load1(concept, lang), start=1)
    ]


def show(data, concept, language):
    data = [
        datum
        for datum in data
        if datum["concept"] == concept and datum["language"] == language
    ]
    data = sorted(data, key=lambda datum: datum["rank"])
    for datum in data:
        audio_path = datasets[language].get_audio_path(datum["filename"])
        st.code(datum)
        st.audio(str(audio_path))
        st.markdown(datum["caption"])
        st.markdown("---")


def main():
    with st.sidebar:
        concept = st.selectbox("concept", datasets["english"].load_concepts())
        language = st.selectbox("language", datasets.keys())

    data = load_data()
    df = pd.DataFrame(data)

    # cumulatively sum the number of correct predictions
    dfg = df.groupby(["concept", "language"])
    df["precision"] = 100 * dfg["has-concept"].cumsum() / df["rank"]

    df_tail = df[df["language"] == "english"].groupby(["concept"]).tail(1)
    df_tail

    df["language"] = df["language"].replace("yoruba", "yorùbá")
    df["language"] = df["language"].str.capitalize()

    fig = sns.relplot(
        data=df, x="rank", y="precision", hue="concept", col="language", kind="line"
    )
    fig.set_titles("{col_name}")
    fig.set(xlabel="Rank", ylabel="Precision (%)")
    fig._legend.set_title("Keyword")
    # sns.lineplot(data=df, x="score", y="precision", hue="concept", ax=axs[1])
    # sns.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))
    # fig.tight_layout()
    st.pyplot(fig)
    # path = "output/taslp/mining-precision-flickr-audio-en-yo.png"
    # fig.savefig(path, dpi=300)
    path = "output/taslp/mining-precision-flickr-audio-en-yo.pdf"
    fig.savefig(path)

    # show(data, concept, language)

    def get_iou(segment1, segment2):
        start1, end1 = segment1["time-start"], segment1["time-end"]
        start2, end2 = segment2["time-start"], segment2["time-end"]
        if start1 > end2 or end1 < start2:
            return 0
        else:
            inter = min(end1, end2) - max(start1, start2)
            union = max(end1, end2) - min(start1, start2)
            return inter / union

    def has_concept_aligned(datum, τ=0.25):
        try:
            alignment = datasets["english"].alignments[datum["filename"]]
        except KeyError:
            alignment = []
        alignment = [a for a in alignment if a["word"] == datum["concept"]]
        overlaps = [get_iou(a, datum) for a in alignment]
        if not overlaps:
            return False
        else:
            return max(overlaps, default=0) >= τ

    data_en = [
        assoc(datum, "has-concept-aligned", has_concept_aligned(datum))
        for datum in data
        if datum["language"] == "english"
    ]
    df = pd.DataFrame(data_en)
    dfg = df.groupby(["concept"])
    df["precision"] = 100 * dfg["has-concept-aligned"].cumsum() / df["rank"]

    df_tail = df.groupby(["concept"]).tail(1)
    df_tail

    fig, axs = plt.subplots(ncols=1, squeeze=False)
    sns.lineplot(data=df, x="rank", y="precision", hue="concept", ax=axs[0, 0])
    sns.move_legend(axs[0, 0], "upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    st.pyplot(fig)

    def get_all_words(datum):
        try:
            caption = datasets[datum["language"]].captions[datum["filename"]]
            return caption.lower().split()
        except KeyError:
            return []

    def get_predicted_word(datum):
        try:
            alignment = datasets["english"].alignments[datum["filename"]]
        except KeyError:
            return None
        try:
            return max(alignment, key=lambda a: get_iou(a, datum))["word"]
        except:
            pdb.set_trace()

    confusions = [
        assoc(datum, "word", predicted_word)
        for datum in data
        if datum["language"] == "english"
        and (predicted_word := get_predicted_word(datum))
    ]
    for concept, group in groupby(confusions, lambda x: x["concept"]):
        group = list(group)
        counter = Counter([x["word"] for x in group])
        counter_str = [f"{word} {count}" for word, count in counter.most_common(5)]
        counter_str = ", ".join(counter_str)
        st.write(concept)
        st.markdown(counter_str)
        st.markdown("---")

    st.markdown("### Co-occurences against any word in the caption")
    confusions = [
        assoc(datum, "word", word) for datum in data for word in get_all_words(datum)
    ]
    for language in LANGUAGES:
        confusions_lang = [
            datum for datum in confusions if datum["language"] == language
        ]
        st.write(language)
        for concept, group in groupby(confusions_lang, lambda x: x["concept"]):
            group = list(group)
            counter = Counter([x["word"] for x in group])
            if language == "yoruba":
                concept_str = "{} ({})".format(
                    datasets[language].concept_to_yoruba[concept],
                    concept,
                )
            else:
                concept_str = concept
            st.write(concept_str)
            st.write(counter.most_common(20))
            st.markdown("---")


if __name__ == "__main__":
    main()
