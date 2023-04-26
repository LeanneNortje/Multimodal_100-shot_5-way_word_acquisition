import os
import pdb
import pandas as pd

from pathlib import Path
from toolz import concat

from predict import load_concepts
from evaluate import Results


episode = 0
concepts = load_concepts()
results = Results()


def load1(concept):
    data = results.load(concept, episode)
    data = sorted(data, reverse=True, key=lambda datum: datum["score"])
    data = data[:5]

    template_ii = lambda n: "\ii{" + str(n) + "}"
    template_img = (
        lambda path: "\includegraphics[width=\imgsize, height=\imgsize]{" + path + "}"
    )

    for rank, datum in enumerate(data, start=1):
        datum["rank"] = template_ii(rank)
        datum["concept"] = concept
        image_name = Path(datum["image-file"]).stem
        datum["image"] = template_img(os.path.join("imgs", image_name + ".jpg"))
        datum["explanation"] = template_img(
            os.path.join("imgs", image_name + "-explanation.jpg")
        )

    return data


columns = ["rank", "concept", "image", "explanation"]
table = pd.DataFrame(list(concat(load1(concept) for concept in concepts)))
table = table[columns]
table = table.set_index(["rank", "concept"])
table = table.unstack(-1).swaplevel(axis="columns")
table = table.reindex(concepts, axis=1, level=0)

with pd.option_context("max_colwidth", 1000):
    table.to_latex(
        "output/taslp/qualitative-results.tex",
        multicolumn=True,
        escape=False,
        index=False,
    )
