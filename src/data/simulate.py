import numpy as np
import pandas as pd


def generate_cat_random(ratio: pd.Series, size: int) -> np.ndarray | None:
    if not isinstance(ratio, pd.Series):
        return None

    return np.random.choice(a=ratio.index, size=size, p=ratio / 100)
