from itertools import combinations
from string import ascii_lowercase

import pandas as pd


def add_psychometrics_features(df: pd.DataFrame) -> pd.DataFrame:
    questions = [alphabet for alphabet in list(ascii_lowercase)[:20]]
    answers = [(f"Q{question}A") for question in questions]

    df["T"] = df["QcA"] - df["QfA"] + df["QoA"] - df["QrA"] + df["QsA"]
    df["V"] = df["QbA"] - df["QeA"] + df["QhA"] + df["QjA"] + df["QmA"] - df["QqA"]
    df["M"] = -df["QkA"]

    flipping_columns = ["QeA", "QfA", "QkA", "QqA", "QrA"]
    for filp in flipping_columns:
        df[filp] = 6 - df[filp]

    flipping_secret_columns = ["QaA", "QdA", "QgA", "QiA", "QnA"]
    for filp in flipping_secret_columns:
        df[filp] = 6 - df[filp]

    df["Math_score"] = df[answers].mean(axis=1)
    df["delay"] = df[[(f"Q{question}E") for question in questions]].sum(axis=1)
    df["delay"] = df["delay"] ** (1 / 10)

    anscoms = list(combinations(answers, 2))
    for com1, com2 in anscoms:
        df[f"{com1}_dv_{com2}"] = df[com1] / df[com2]

    df = df.drop(columns=[(f"Q{question}A") for question in questions])
    df = df.drop(columns=[(f"Q{question}E") for question in questions])
    df = df.drop(columns=["hand"])

    wr_list = [("wr_" + str(i).zfill(2)) for i in range(1, 14)]
    wr_no_need = [i for i in wr_list if i not in ["wr_01", "wr_03", "wr_06", "wr_09", "wr_11"]]
    df = df.drop(columns=wr_no_need)

    df["Ex"] = df["tp01"] - df["tp06"]
    df["Ag"] = df["tp07"] - df["tp02"]
    df["Con"] = df["tp03"] - df["tp08"]
    df["Es"] = df["tp09"] - df["tp04"]
    df["Op"] = df["tp05"] - df["tp10"]

    df = df.drop(columns=[("tp0" + str(i)) for i in range(1, 10)])

    return df
