from typing import List, TypedDict

import pdb

import pandas as pd
import seaborn as sns
import streamlit as st

# data taken from paper
tables_str = {
    "few-shot retrieval": r"""K & brocolli & fire hydrant & kite & sheep & zebra\\
5 & 40.4 & 11.3 & 27.5 & 33.3 & 77.8\\
10 & 49.1 & 9.7 & 30.8 & 33.3 & 85.6\\
50 & 50.9 & 8.1 & 20.9 & 36.5 & 82.2\\
100 & 52.6 & 4.8 & 33.0 & 38.1 & 80.0\\""",
    "audio mining": r"""K & brocolli & fire hydrant & kite & sheep & zebra\\
5 & 95.4 & 97.7 & 57.0 & 63.3 & 91.9\\
10 & 90.5 & 97.7 & 57.5 & 70.7 & 98.7\\
50 & 85.9 & 97.5 & 51.7 & 93.1 & 98.9\\
100 & 93.9 & 97.6 & 54.7 & 92.4 & 99.4\\""",
    "image mining": r"""K & brocolli & fire hydrant & kite & sheep & zebra\\
5 & 53.1 & 6.0 & 38.8 & 38.2 & 83.6\\
10 & 59.7 & 6.7 & 45.1 & 38.5 & 87.5\\
50 & 60.3 & 16.15 & 32.8 & 39.9 & 90.9\\
100 & 61.0 & 21.9 & 39.5 & 41.0 & 94.0\\""",
}

Ks = [0, 5, 10, 50, 100]


def mapt(f, tup):
    return tuple(map(f, tup))


def parse_table(name, table_str):
    def parse_header(header):
        columns = header.split("&")
        columns = mapt(str.strip, columns)
        return columns

    def parse_row(row):
        k, *scores = row.split("&")
        return int(k), *mapt(float, scores)

    K_TO_IDX = {k: i for i, k in enumerate(Ks)}

    header, *rows = table_str.split(r"\\")
    columns = parse_header(header)
    data = [parse_row(row) for row in rows if row]
    df = pd.DataFrame(data, columns=columns)
    df = df.set_index("K")
    df = df.unstack().reset_index()
    df = df.rename(columns={"level_0": "keyword", 0: "precision"})
    df["task"] = name
    df["K"] = df["K"].map(K_TO_IDX)
    return df


def parse_all(tables_str):
    df = pd.concat([parse_table(k, t) for k, t in tables_str.items()])
    return df


st.set_page_config(layout="wide")

df = parse_all(tables_str)
df

sns.set_context("talk")
fig = sns.relplot(
    data=df,
    x="K",
    y="precision",
    hue="task",
    style="task",
    col="keyword",
    kind="line",
    markers=True,
)
fig.set(xticklabels=mapt(str, Ks))

st.pyplot(fig)
