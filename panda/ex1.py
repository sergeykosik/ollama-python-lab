# %%
# Example 1: Import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


df = pd.read_csv("titanic.csv")

# %% Example 2: Show data
df.head()
# %%
df.describe()
# %%
df.describe().T
# %%
df.shape
# %%
df.columns
# %%
df.isnull().sum()
# %%
df_null_data = df[df.isnull().any(axis=1)]
df_null_data.head(10)

# %%
# Example 3: Impuding data

df['country'].value_counts()
df.country.mode()[0]
# %%
df["country"].fillna('England', inplace=True)
# %%
df["age"].plot.hist(bins=100)
# %%
