import pdb
import json

from pathlib import Path

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
# from sklearn.metrics import top_k_accuracy_score

from predict import BASE_DIR, IMAGE_COCO_DIR, load_concepts


class Results:
    def __init__(self):
        self.coco = COCO(IMAGE_COCO_DIR / "annotations" / "instances_val2014.json")
        self.episodes = np.load("data/test_episodes.npz", allow_pickle=True)
        self.episodes = self.episodes["episodes"].item()
        self.image_matching_set = sorted(self.episodes["matching_set"].keys())

    def get_coco_annots(self, image_file, concept):
        image_name = Path(image_file).stem
        coco_image_id = [int(image_name.split("_")[-1])]
        coco_category_ids = self.coco.getCatIds(catNms=[concept])
        coco_annot_ids = self.coco.getAnnIds(
            imgIds=coco_image_id, catIds=coco_category_ids
        )
        coco_annots = self.coco.loadAnns(coco_annot_ids)
        return coco_annots

    def load(self, concept, episode):
        concept_str = concept.replace(" ", "-")
        scores = np.load(f"data/scores/{concept_str}-{episode}.npy")

        def is_query_in_caption(image_file):
            return concept in self.episodes["matching_set"][image_file]

        def is_query_in_image(image_file):
            return len(self.get_coco_annots(image_file, concept)) > 0

        return [
            {
                "score": scores[i],
                "image-file": image_file,
                "is-query-in-caption": is_query_in_caption(image_file),
                "is-query-in-image": is_query_in_image(image_file),
            }
            for i, image_file in enumerate(self.image_matching_set)
        ]


def evaluate_p_at_n(results, label_type):
    true = np.array([r[label_type] for r in results])
    pred = np.array([r["score"] for r in results])
    n = sum(true)
    idxs = np.argsort(-pred)
    idxs = idxs[:n]
    return sum(true[idxs]) / n


def main():
    concepts = load_concepts()
    results = Results()
    metrics = [
        {
            "concept": concept,
            "episode": episode,
            "p@n": evaluate_p_at_n(
                results.load(concept, episode),
                label_type="is-query-in-caption",
            ),
        }
        for concept in concepts
        for episode in range(100)
    ]
    df = pd.DataFrame(metrics)
    print(df.groupby("concept").mean())
    pdb.set_trace()


if __name__ == "__main__":
    main()
