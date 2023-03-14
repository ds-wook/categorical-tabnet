# %%
import pandas as pd
from catboost.datasets import epsilon

epsilon_train, epsilon_test = epsilon()

# %%
epsilon_train.head()

# %%
train = pd.read_csv("../input/shrutime/train.csv")
test = pd.read_csv("../input/shrutime/test.csv")

# %%
train.head()
# %%
train.tail()
# %%
