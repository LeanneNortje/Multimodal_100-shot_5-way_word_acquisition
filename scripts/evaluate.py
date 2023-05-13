import pdb
import random

from functools import reduce
from pathlib import Path
from tqdm import tqdm

import click
import numpy as np
import pandas as pd

from pycocotools.coco import COCO

# from sklearn.metrics import top_k_accuracy_score

from predict import IMAGE_COCO_DIR, load_concepts


class Results:
    def __init__(self, config_name):
        self.config_name = config_name
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

    def load_retrieval(self, concept, episode):
        concept_str = concept.replace(" ", "-")
        path = f"data/scores-{self.config_name}/{concept_str}-{episode}.npy"
        scores = np.load(path)

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

    def load_classification(self, concept, episode):
        concept_str = concept.replace(" ", "-")
        path = f"data/scores-{self.config_name}/{concept_str}-{episode}.npy"
        scores = np.load(path)
        matching_set = self.episodes[episode]["matching_set"]

        def is_query_in_caption(image_file):
            return concept in self.episodes["matching_set"][image_file]

        def is_query_in_image(image_file):
            return len(self.get_coco_annots(image_file, concept)) > 0

        return [
            {
                "score": scores[self.image_matching_set.index(image_file)],
                "image-file": image_file,
                "image-concept": category,
                "is-query-in-caption": is_query_in_caption(image_file),
                "is-query-in-image": is_query_in_image(image_file),
            }
            for category, image_file in matching_set.items()
        ]

    load = load_retrieval


def evaluate_classification(
    concepts, results, label_type="is-query-in-caption", *, episode
):
    # FIXME This is evaluation doesn't match what's done in the paper.
    # TODO Use the right support set!
    data = [
        {**r, "concept": concept}
        for concept in concepts
        for r in results.load(concept, episode)
    ]
    df = pd.DataFrame(data)

    true = df[df[label_type]][["image-file", "concept"]].set_index("image-file")
    pred = df.pivot("image-file", "concept", "score").idxmax(1)

    true_pred = true.join(pred.to_frame("pred"))
    # num_match = (true_pred["concept"] == true_pred["pred"]).sum()
    # num_total = len(true_pred)
    # return 100 * num_match / num_total

    is_correct = true_pred["concept"] == true_pred["pred"]
    true_pred = true_pred.join(is_correct.to_frame("is-correct"))
    counts = true_pred.groupby("concept")["is-correct"].agg(
        num_match="sum", num_total="size"
    )

    return counts


def evaluate_retrieval(results, label_type):
    true = np.array([r[label_type] for r in results])
    pred = np.array([r["score"] for r in results])
    n = sum(true)
    idxs = np.argsort(-pred)
    idxs = idxs[:n]
    return 100 * sum(true[idxs]) / n


@click.command()
@click.option("-c", "--config", "config_name", required=True)
def main(config_name):
    NUM_EPISODES = 10

    concepts = load_concepts()
    results = Results(config_name)

    counts = [
        # evaluate_classification(concepts, results, episode=random.randint(0, 999))
        evaluate_classification(concepts, results, episode=episode)
        for episode in tqdm(range(NUM_EPISODES))
    ]
    counts = reduce(lambda x, y: x.add(y), counts)
    # print("accuracy")
    print(100 * counts["num_match"] / counts["num_total"])
    # print("accuracy: {:.2f}±{:.1f}".format(np.mean(accs), np.std(accs)))

    metrics = [
        {
            "concept": concept,
            "episode": episode,
            "metric": evaluate_retrieval(
                results.load(concept, episode),
                label_type="is-query-in-caption",
            ),
        }
        for concept in concepts
        for episode in range(NUM_EPISODES)
    ]
    df = pd.DataFrame(metrics)
    print(df.groupby("concept").mean())
    print(df["metric"].mean())
    pdb.set_trace()


if __name__ == "__main__":
    main()
