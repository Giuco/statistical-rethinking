import pandas as pd
import numpy as np


def simplehist(x: np.ndarray):
    return pd.Series(x).value_counts().sort_index().plot.bar()
