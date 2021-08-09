import pandas as pd
import numpy as np
from causalgraphicalmodels import CausalGraphicalModel
from typing import Optional, List, Set
from scipy import stats
from itertools import product


def lfilter(*args, **kwargs):
    return list(filter(*args, **kwargs))


def posterior_grid_approx(grid_points: Optional[int] = 100,
                          success: Optional[int] = 6, 
                          tosses: Optional[int] = 9,
                          prior: Optional[str] = 'uniform'):
    # define grid
    p_grid = np.linspace(0, 1, grid_points)

    # define prior
    if prior == 'uniform':
        prior = np.repeat(5, grid_points)  # uniform
    elif prior == 'truncated':
        prior = (p_grid >= 0.5).astype(int)  # truncated
    elif prior == 'double_exp':
        prior = np.exp(- 5 * abs(p_grid - 0.5))  # double exp
    else:
        raise ValueError("prior not valid")
        
    # compute likelihood at each point in the grid
    likelihood = stats.binom.pmf(success, tosses, p_grid)

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()
    return p_grid, posterior


def simplehist(x: np.ndarray):
    return pd.Series(x).value_counts().sort_index().plot.bar()


def create_standardizer_destandardizer(s: np.ndarray) -> np.ndarray:
    return lambda x: (x - s.mean()) / s.std(), lambda x: (x * s.std()) + s.mean()


class CausalModel(CausalGraphicalModel):
    def get_implied_conditional_independencies(self):
        all_independencies = self.get_all_independence_relationships()
        conditional_independencies = []
        for s in all_independencies:
            if all(
                t[0] != s[0] or t[1] != s[1] or not t[2].issubset(s[2])
                for t in all_independencies
                if t != s
            ):
                conditional_independencies.append(s)
                
        return conditional_independencies
    
    def get_adjustment_sets(self, x: str, y: str) -> List[Set]:
        all_adjustment_sets = self.get_all_backdoor_adjustment_sets(x, y)
        filtered_adjustment_sets = []
        for a in all_adjustment_sets:
            if all(not b.issubset(a) for b in all_adjustment_sets if a != b):
                filtered_adjustment_sets.append(a)

        return filtered_adjustment_sets
