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

from predict import CONFIGS, COCOData, FlickrEnData, FlickrYoData


class Results:
    def __init__(self, config_name, dataset):
        LOADERS = {
            "retrieval": self.load_retrieval,
            "classification": self.load_classification,
        }
        config = CONFIGS[config_name]
        task = config["task"]

        self.dataset = dataset
        self.config_name = config_name
        self.load = LOADERS[task]

    def _load_scores(self, concept, episode):
        concept_str = concept.replace(" ", "-")
        path = f"data/scores-{self.config_name}/{concept_str}-{episode}.npy"
        scores = np.load(path)
        return scores

    def is_query_in_caption(self, concept, image_file):
        concept_lang = self.dataset.translate_concept(concept)
        captions = self.dataset.captions_image[image_file]
        words = " ".join(captions).lower().split()
        return any(concept_lang == word for word in words)

    def is_query_in_image(self, concept, image_file):
        raise NotImplementedError

    def load_retrieval(self, concept, episode):
        scores = self._load_scores(concept, episode)
        return [
            {
                "score": scores[i],
                "image-file": image_file,
                "is-query-in-caption": self.is_query_in_caption(concept, image_file),
                "is-query-in-image": self.is_query_in_image(concept, image_file),
            }
            for i, image_file in enumerate(self.dataset.image_matching_set)
        ]

    def load_classification(self, concept, episode):
        try:
            scores = self._load_scores(concept, episode)
        except FileNotFoundError:
            return []

        episode_matching_set = self.dataset.episodes[episode]["matching_set"]
        episode_matching_set_images = list(episode_matching_set.values())

        return [
            {
                "score": scores[episode_matching_set_images.index(image_file)],
                "image-file": image_file,
                "image-concept": category,
                "is-query": category == concept,  # TODO Use this as "label-type"
                "is-query-in-caption": self.is_query_in_caption(concept, image_file),
                "is-query-in-image": self.is_query_in_image(concept, image_file),
            }
            for category, image_file in episode_matching_set.items()
        ]


class COCOResults(Results):
    def __init__(self, config_name, dataset, split="val2014"):
        super().__init__(config_name, dataset)
        self.coco = COCO(dataset.image_dir / "annotations" / f"instances_{split}.json")
        self.split = split
        # self.load = self.load_retrieval

    def get_coco_annots(self, image_file, concept):
        image_file = Path(image_file)
        folder, _ = image_file.parts
        assert folder == self.split
        image_name = image_file.stem
        coco_image_id = [int(image_name.split("_")[-1])]
        coco_category_ids = self.coco.getCatIds(catNms=[concept])
        coco_annot_ids = self.coco.getAnnIds(
            imgIds=coco_image_id, catIds=coco_category_ids
        )
        coco_annots = self.coco.loadAnns(coco_annot_ids)
        return coco_annots

    def is_query_in_image(self, concept, image_file):
        return len(self.get_coco_annots(image_file, concept)) > 0


class FlickrEnResults(Results):
    def __init__(self, config_name, dataset):
        super().__init__(config_name, dataset)

    def is_query_in_image(self, concept, image_file):
        return None


class FlickrYoResults(Results):
    def __init__(self, config_name, dataset):
        super().__init__(config_name, dataset)

    def is_query_in_image(self, concept, image_file):
        return None

    def is_query_in_caption(self, concept, image_file):
        concept_lang = self.dataset.translate_concept(concept)
        alignment = self.dataset.alignments[image_file + "_0"]
        words = [a["word"].lower() for a in alignment]
        return concept_lang in words


def evaluate_classification(
    concepts, results, label_type="is-query-in-caption", *, episode
):
    # FIXME This is evaluation doesn't match what's done in the paper.
    # TODO Use the right support set!
    data = [
        {**r, "concept": concept}
        for concept in concepts
        for r in results.load_classification(concept, episode)
    ]
    df = pd.DataFrame(data)

    true = df[df[label_type]].groupby("image-file")["concept"].agg(lambda s: s.tolist())
    true = true.to_frame("true-concepts").reset_index()

    pred_cols = ["image-file", "concept", "score"]
    pred = df[pred_cols].drop_duplicates().pivot(*pred_cols).idxmax(0)
    pred = pred.to_frame("pred-image-file").reset_index()
    # try:
    #     # true = df[df[label_type]][["image-file", "concept"]].set_index("image-file")
    # except:
    #     return None

    true_pred = pred.merge(
        right=true, left_on="pred-image-file", right_on="image-file", how="left"
    )
    true_pred["is-correct"] = 100 * true_pred.apply(
        lambda x: isinstance(x["true-concepts"], list)
        and x["concept"] in x["true-concepts"],
        axis=1,
    )

    # true_pred = true.join(pred.to_frame("pred"))
    # num_match = (true_pred["concept"] == true_pred["pred"]).sum()
    # num_total = len(true_pred)
    # return 100 * num_match / num_total

    # true_pred["is-correct"] = true_pred["concept"] == true_pred["pred"]
    # counts = true_pred.groupby("concept")["is-correct"].agg(
    #     num_match="sum", num_total="size"
    # )

    return dict(true_pred[["concept", "is-correct"]].values.tolist())


def evaluate_retrieval(results, label_type):
    true = np.array([r[label_type] for r in results])
    pred = np.array([r["score"] for r in results])
    n = sum(true)
    idxs = np.argsort(-pred)
    idxs = idxs[:n]
    return 100 * sum(true[idxs]) / n


EVALUATORS = {
    "retrieval": evaluate_retrieval,
    "classification": evaluate_classification,
}


@click.command()
@click.option(
    "-c", "--config", "config_name", required=True, type=click.Choice(CONFIGS)
)
def main(config_name):
    NUM_EPISODES = 1000

    config = CONFIGS[config_name]
    dataset = config["data-class"]()
    task = config["task"]

    if config["data-class"] == COCOData:
        Results = COCOResults
    elif config["data-class"] == FlickrEnData:
        Results = FlickrEnResults
    elif config["data-class"] == FlickrYoData:
        Results = FlickrYoResults
    else:
        raise ValueError(config["data-class"])

    concepts = dataset.load_concepts()
    results = Results(config_name, dataset)
    evaluate = EVALUATORS[task]

    counts = [
        # evaluate(concepts, results, episode=random.randint(0, 999))
        evaluate(concepts, results, episode=episode)
        for episode in tqdm(range(NUM_EPISODES))
    ]

    def compute_average(counts, concept):
        counts1 = [counts.get(concept, None) for counts in counts]
        counts1 = [count for count in counts1 if count is not None]
        return np.mean(counts1)

    results = {concept: compute_average(counts, concept) for concept in concepts}
    print(results)
    # counts = [c for c in counts if c is not None]
    # print(len(counts))
    # counts = reduce(lambda x, y: x.add(y), counts)
    # # print("accuracy")
    # print(100 * counts["num_match"] / counts["num_total"])
    # print("accuracy: {:.2f}±{:.1f}".format(np.mean(accs), np.std(accs)))

    # metrics = [
    #     {
    #         "concept": concept,
    #         "episode": episode,
    #         "metric": evaluate_retrieval(
    #             results.load(concept, episode),
    #             label_type="is-query-in-caption",
    #         ),
    #     }
    #     for concept in concepts
    #     for episode in range(NUM_EPISODES)
    # ]
    # df = pd.DataFrame(metrics)
    # print(df.groupby("concept").mean())
    # print(df["metric"].mean())
    # pdb.set_trace()


if __name__ == "__main__":
    main()
