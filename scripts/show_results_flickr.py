import pdb
import shutil

from pathlib import Path

import librosa
import numpy as np
import seaborn as sns
import streamlit as st
import torch

from matplotlib import pyplot as plt
from torch import nn
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

from predict import FlickrEnData, FlickrYoData, MattNet
from evaluate import FlickrEnResults, FlickrYoResults


st.set_page_config(layout="wide")


CONFIG_EN = "flickr-en-5-cls"
CONFIG_YO = "flickr-yo-5-pretrained-cls"

TO_SAVE_DATA_FOR_PAPER = True
PAPER_DIR = Path("output/taslp/imgs/flickr-new")


datasets = {
    "english": FlickrEnData(),
    "yoruba": FlickrYoData(),
}

mattnets = {
    "english": MattNet(CONFIG_EN),
    "yoruba": MattNet(CONFIG_YO),
}

results = {
    "english": FlickrEnResults(CONFIG_EN, datasets["english"]),
    "yoruba": FlickrYoResults(CONFIG_YO, datasets["yoruba"]),
}


class MattNetForGradCAM(nn.Module):
    def __init__(self, mattnet, audio):
        super().__init__()
        self.mattnet = mattnet
        self.audio = audio

    def forward(self, image):
        score, _ = self.mattnet(self.audio, image)
        return [score]


def make_image_square(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    return image


# with st.sidebar:
#     episode_no = st.number_input(
#         "episode no.", min_value=0, max_value=1000, format="%d", step=1
#     )


def show_matching_set(episode_no, language="english"):
    concepts = datasets[language].load_concepts()
    episode_info = datasets[language].episodes[episode_no]
    matching_set = datasets[language].episodes[episode_no]["matching_set"]
    num_matching_set = len(matching_set)
    st.markdown("#### matching set")
    cols = st.columns(num_matching_set)
    for i, (concept, key) in enumerate(matching_set.items()):
        image_path = datasets["english"].get_image_path(key)
        concept_en = datasets[language].back_translate_concept(concept)
        concepts_leanne_str = ", ".join(episode_info["possible_words"][concept])
        concepts_captions = [concept for concept in concepts if results[language].is_query_in_caption(concept, key)]
        concepts_captions_str = ", ".join(concepts_captions)
        cols[i].markdown("labels leanne: " + concepts_leanne_str)
        cols[i].markdown("labels captions: " + concepts_captions_str)
        cols[i].image(make_image_square(str(image_path)), caption=image_path.stem)
        if TO_SAVE_DATA_FOR_PAPER:
            shutil.copy(
                image_path, PAPER_DIR / f"matching-set-{language}-{episode_no:03d}-{concept_en}.jpg"
            )
    st.markdown("---")


def show_predictions(*, episode_no, language):
    sample_rate = 16_000
    window_stride = 0.01

    # def load_scores(concept):
    #     lang_code = language[:2]
    #     concept_str = concept.replace(" ", "-")
    #     path = "data/scores-flickr-{}-5-cls/{}-{}.npy".format(
    #         lang_code,
    #         concept_str,
    #         episode_no,
    #     )
    #     return np.load(path)

    def get_audio_for_concept(concept, key):
        # Move logic to dataset
        if language == "yoruba":
            key1 = datasets[language].trim_prefix(key)
            sample_rate = 48_000
        else:
            key1 = key
            sample_rate = 16_000
        audio_path = str(datasets[language].get_audio_path(key1))
        y, _ = librosa.load(audio_path, sr=sample_rate)

        k = window_stride * sample_rate
        α, ω = datasets[language].get_alignment_episode_concept(episode_no, concept)
        α = int(k * α)
        ω = int(k * ω)
        y = y[α:ω]

        return y

    def plot_audio(y):
        fig, ax = plt.subplots()
        ax.plot(y)
        ax.axis("off")
        return fig

    def get_explanation_image(concept, image_key, col=None):
        audio_path = datasets[language].get_audio_path_episode_concept(
            episode_no, concept
        )
        alignment = datasets[language].get_alignment_episode_concept(
            episode_no, concept
        )
        audio = mattnets[language].load_audio_1(audio_path, alignment)

        # if language == "yoruba":
        #     st.write(concept)
        #     st.write(alignment)
        #     pdb.set_trace()

        image_path = str(datasets[language].get_image_path(image_key))
        image = mattnets[language].load_image_1(image_path)
        image = image.unsqueeze(0)

        # original image
        image_rgb = Image.open(image_path)
        image_rgb = image_rgb.convert("RGB")
        # image_rgb = image_rgb.resize((w, h))
        image_rgb = np.array(image_rgb) / 255
        h, w, _ = image_rgb.shape

        # mattnet_for_gradcam = MattNetForGradCAM(mattnets[language], audio)
        # grad_cam = GradCAM(
        #     model=mattnet_for_gradcam,
        #     target_layers=[mattnet_for_gradcam.mattnet.image_model[-1]],
        # )
        # targets = [RawScoresOutputTarget()]

        # explanation = grad_cam(input_tensor=image, targets=targets)[0]

        with torch.no_grad():
            score, attention = mattnets[language](audio, image)

        # if image_key == "2748729903_3c7c920c4d":
        #     pdb.set_trace()
        attention = attention.view(7, 7)
        attention = 5 * (attention / 100 - 0.5)
        explanation = torch.sigmoid(attention).numpy()

        explanation = Image.fromarray(explanation)
        explanation = explanation.resize((w, h))
        explanation = np.array(explanation)

        image_explanation = show_cam_on_image(image_rgb, explanation, use_rgb=True)
        image_explanation = Image.fromarray(image_explanation)
        # image_explanation = image_explanation.resize((224, 224))
        return np.array(image_explanation)

    matching_set = datasets[language].episodes[episode_no]["matching_set"]
    matching_set_concepts = list(matching_set.keys())
    matching_set_images = list(matching_set.values())
    image_to_concept = {v: k for k, v in matching_set.items()}
    episode_info = datasets[language].episodes[episode_no]
    queries = datasets[language].episodes[episode_no]["queries"]
    num_queries = len(queries)
    st.markdown("#### predictions")
    cols = st.columns(num_queries)

    # entire confusion matrix
    # for i, (concept_true, key) in enumerate(queries.items()):
    #     try:
    #         scores = load_scores(concept_true)
    #     except:
    #         continue

    #     for j, concept_pred in enumerate(matching_set_concepts):
    #         image_pred_key = matching_set_images[j]
    #         cols[i].image(get_explanation_image(concept_true, image_pred_key, cols[i]))
    #         cols[i].markdown("{} · {:.2f}".format(concept_pred, scores[j]))

    num_captions = 5 if language == "english" else 1

    for i, (concept, key) in enumerate(queries.items()):
        concept_en = datasets[language].back_translate_concept(concept)
        scores = results[language]._load_scores(concept_en, episode_no)

        idx = np.argmax(scores)
        audio_concept = get_audio_for_concept(concept_en, key)
        image_pred_key = matching_set_images[idx]
        # concept_pred = matching_set_concepts[idx]
        concept_image = image_to_concept[image_pred_key]
        is_correct_leanne = concept in episode_info["possible_words"][concept_image]
        is_correct_captions = results[language].is_query_in_caption(
            concept_en, image_pred_key
        )
        is_correct_leanne_str = "✅" if is_correct_leanne else "❌"
        is_correct_captions_str = "✅" if is_correct_captions else "❌"
        # try:
        #     assert is_correct_captions == is_correct_leanne
        # except:
        #     pdb.set_trace()
        fig_audio = plot_audio(audio_concept)
        image_explanation = get_explanation_image(concept_en, image_pred_key, cols[i])
        captions = [
            datasets[language].captions[image_pred_key + "_" + str(i)]
            for i in range(num_captions)
        ]

        cols[i].markdown("query audio: {}".format(concept_en))
        cols[i].audio(audio_concept, sample_rate=sample_rate)
        cols[i].pyplot(fig_audio)
        cols[i].markdown("`{}`".format(queries[concept]))
        cols[i].markdown("predicted image + attention:")
        cols[i].image(image_explanation)
        cols[i].markdown(
            "{:.2f} · captions: {} · leanne: {}".format(
                scores[idx], is_correct_captions_str, is_correct_leanne_str
            )
        )
        cols[i].markdown("all scores:")
        cols[i].write(scores)
        cols[i].markdown("captions:")
        cols[i].write(captions)

        if TO_SAVE_DATA_FOR_PAPER:
            fig_audio.savefig(
                PAPER_DIR / f"audio-{episode_no:03d}-{concept_en}-{language}.png",
                bbox_inches="tight",
                pad_inches=0,
            )
            im = Image.fromarray(image_explanation)
            im.save(
                PAPER_DIR / f"explanation-{episode_no:03d}-{concept_en}-{language}.png"
            )

    st.markdown("---")


def main():
    for e in range(1):
        for l in ("english", "yoruba"):
            st.markdown("### episode: {} · language: {}".format(e, l))
            show_matching_set(episode_no=e, language=l)
            show_predictions(episode_no=e, language=l)


if __name__ == "__main__":
    main()
