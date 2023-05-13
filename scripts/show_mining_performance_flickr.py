from typing import List, TypedDict

import pdb
import re

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from predict import FlickrEnData, FlickrYoData
from utils import load

sns.set_context("talk")

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


def load_data() -> List[Datum]:
    def load1(concept, lang):
        folder = datasets["english"].base_metadata_dir
        path = (
            folder
            / ("super_5-shot_5-way_" + lang)
            / "data"
            / "audio_pair_lists"
            / (concept + ".txt")
        )
        return load(path, lambda line: line.strip())

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
        }
        for concept in concepts
        for lang in languages
        for rank, filename in enumerate(load1(concept, lang), start=1)
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

    fig = sns.relplot(
        data=df, x="rank", y="precision", hue="concept", col="language", kind="line"
    )
    # sns.lineplot(data=df, x="score", y="precision", hue="concept", ax=axs[1])
    # sns.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))
    # fig.tight_layout()
    st.pyplot(fig)

    show(data, concept, language)


if __name__ == "__main__":
    main()
