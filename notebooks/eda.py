# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("../input/shrutime/train.csv")
df.head()
# %%
X = {}
y = {}
X["train"], X["test"], y["train"], y["test"] = train_test_split(
    df.drop(["Exited"], axis=1), df["Exited"], train_size=0.8
)

# %%
X["train"]
# %%
