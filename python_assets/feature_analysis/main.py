import statsmodels.api as sm
import pandas as pd
import shared_vars as var
import matplotlib.pyplot as plt
import numpy as np
import logging

LOGGER = logging.getLogger("Log")

df = pd.read_csv(var.DATA)
df['constant'] = 1.0
train_cols = [c for c in df.columns if c != 'class']
print(df[df['class'] == 1].shape[0] / df.shape[0])

LOGGER.warning('POS crosstab')
print(pd.crosstab(df['same_POS'], df['class']))


print(train_cols)
logit = sm.Logit(df['class'], df[train_cols])
result = logit.fit()

print(result.summary())
print(np.exp(result.params))