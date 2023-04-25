# %%
import pandas as pd
from sklearn.datasets import make_classification

# %%
df = pd.read_csv("../input/shrutime/train.csv")
df.head()
# %%
df["Age_cat"] = pd.qcut(df["Age"], q=4, labels=False)
df.head()
# %%
df.columns
# %%
df.info()
# %%
num_features = [
    col
    for col in df.columns
    if col not in ["Tenure", "HasCrCard", "IsActiveMember", "IsActiveMember", "Geography", "Gender"]
]

num_features
# %%
import numpy as np

a = np.random.rand()
# %%
a
# %%
