import pdb

import pandas as pd
import seaborn as sns
import streamlit as st

tables_str = {
    "retrieval": r"""model & 5 & 10 & 50 & 100\\
mattnet & 40.3$\pm$0.1 & 44.2$\pm$0.1 & 41.7$\pm$0.2 & 43.7$\pm$0.1\\
no bg data & 18.1$\pm$1.0 & 23.6$\pm$1.5 & 26.1$\pm$0.2 & 23.1$\pm$0.5\\
no bg images & 29.9$\pm$0.1 & 31.4$\pm$0.2 & 32.2$\pm$0.2 & 32.1$\pm$0.1\\""",
    "classification": r"""model & 5 & 10 & 50 & 100\\
mattnet & 80.1 & 81.1 & 88.5 & 93.2\\
no bg data & 65.1 & 64.7 & 75.3 & 77.5\\
no bg images & 88.0 & 90.1 & 94.8 & 95.3\\""",
}


def mapt(f, tup):
    return tuple(map(f, tup))


def parse_table(name, table_str):
    def parse_header(header):
        columns = header.split("&")
        columns = mapt(str.strip, columns)
        fst, *rest = columns
        return fst, *mapt(int, rest)

    def parse_row(row):
        columns = row.split("&")
        columns = mapt(str.strip, columns)
        fst, *rest = columns
        rest = mapt(lambda x: x.split("$")[0], rest)
        return fst, *mapt(float, rest)

    header, *rows = table_str.split(r"\\")
    columns = parse_header(header)
    data = [parse_row(row) for row in rows if row]
    df = pd.DataFrame(data, columns=columns)
    df = df.set_index("model")
    df = df.unstack().reset_index()
    df = df.rename(columns={"level_0": "K", 0: "performance"})
    df["task"] = name
    return df


def parse_all(tables_str):
    df = pd.concat([parse_table(k, t) for k, t in tables_str.items()])
    return df

st.set_page_config(layout="wide")

df = parse_all(tables_str)
df

sns.set_context("talk")
df = pd.pivot_table(df, columns="task", index=["K", "model"], values="performance")
df = df.reset_index()
df

fig = sns.relplot(
    data=df,
    x="retrieval",
    y="classification",
    hue="model",
    size="K",
    aspect=1.1,
    # style="task",
)
# sns.move_legend(fig, "lower center", bbox_to_anchor=(1.0, 0.5), ncol=1)
st.pyplot(fig)