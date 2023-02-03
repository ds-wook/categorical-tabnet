from __future__ import annotations

import numpy as np
import pandas as pd


def add_census_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add census features
    Args:
        config: config
        df: dataframe
    Returns:
        dataframe
    """
    df["workclass_occupation"] = df["workclass"] + "#" + df["occupation"]
    df["workclass_education"] = df["workclass"] + "#" + df["education"]
    df["occupation_education"] = df["occupation"] + "#" + df["education"]
    df["marital_status_relationship"] = df["marital_status"] + "#" + df["relationship"]
    df["race_sex"] = df["race"] + "#" + df["sex"]
    df["capital_margin"] = df["capital_gain"] - df["capital_loss"]
    df["capital_total"] = df["capital_gain"] + df["capital_loss"]
    df["capital_margin_flag"] = np.nan
    df.loc[df["capital_margin"] == 0, "capital_margin_flag"] = "zero"
    df.loc[df["capital_margin"] > 0, "capital_margin_flag"] = "positive"
    df.loc[df["capital_margin"] < 0, "capital_margin_flag"] = "negative"

    return df
