import os
import pdb

from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from predict import COCOData, MattNet
from evaluate import COCOResults


config_name = os.environ.get("CONFIG", "100-loc-v2-ret")

dataset = COCOData()
concepts = dataset.load_concepts()

results = COCOResults(config_name, dataset)
mattnet = MattNet(config_name)
mattnet.eval()


def evaluate_localisation_concept(episode_no, query_concept, τ=0.5):
    audio_query, _ = results.dataset.episodes[episode_no]["queries"][query_concept]
    audio_path = dataset.get_audio_path(audio_query)
    audio_name = audio_path.stem

    alignment = dataset.alignments[audio_name][query_concept]
    audio = mattnet.load_audio_1(audio_path, alignment)

    data = results.load(query_concept, episode_no)
    data = [
        datum
        for datum in data
        if results.is_query_in_image(query_concept, datum["image-file"])
    ]

    def load_mask_annotation(datum):
        coco_annots = results.get_coco_annots(datum["image-file"], query_concept)
        assert len(coco_annots) > 0

        masks = [results.coco.annToMask(a) for a in coco_annots]
        masks = np.stack(masks)

        mask = masks.sum(axis=0)
        return mask > 0

    def load_mask_prediction(datum):
        image_file = datum["image-file"]
        image_path = dataset.get_image_path(image_file)
        image_name = image_path.stem

        BASE_DIR = Path(f"output/taslp/evaluate-localisation-preds")
        path_pred = BASE_DIR / f"{config_name}-{episode_no}-{query_concept}" / f"{image_name}.npy"

        path_pred.parent.mkdir(parents=True, exist_ok=True)

        if path_pred.exists():
            pred = np.load(path_pred)
            return pred > τ

        image = mattnet.load_image_1(image_path)
        image = image.unsqueeze(0)

        with torch.no_grad():
            _, attention = mattnet(audio, image)

        attention = attention.view(7, 7)
        attention = 5 * (attention / 100 - 0.5)

        w, h = Image.open(image_path).size

        explanation = torch.sigmoid(attention).numpy()
        explanation = Image.fromarray(explanation).resize((w, h))
        explanation = np.array(explanation)

        np.save(path_pred, explanation)

        return explanation > τ

    def iou(true, pred):
        inter = np.logical_and(true, pred)
        union = np.logical_or(true, pred)
        return np.sum(inter) / np.sum(union)

    def evaluate_datum(datum):
        true = load_mask_annotation(datum)
        pred = load_mask_prediction(datum)
        return iou(true, pred)

    scores = [evaluate_datum(datum) for datum in tqdm(data)]
    return 100 * np.mean(scores)


results = [
    {
        "iou": evaluate_localisation_concept(episode_no, query_concept),
        "query": query_concept,
        "episode-no": episode_no,
    }
    for query_concept in concepts
    for episode_no in range(10)
]

df = pd.DataFrame(results)
print(df.groupby("query")["iou"].agg(["mean", "std"]))