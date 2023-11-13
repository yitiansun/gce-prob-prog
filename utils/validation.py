"""Functions for model validation."""

import numpy as np
import arviz as az
from scipy import special


def find_hdi_prob(samples, value, low=0, high=1, level=15):
    """Recursively find probability for which the hdi have value on boundary."""
    mid = (low + high) / 2
    if level == 0:
        return mid
    hdi = az.hdi(samples, mid)
    if hdi[0] <= value and value <= hdi[1]:
        return find_hdi_prob(samples, value, low, mid, level-1)
    else:
        return find_hdi_prob(samples, value, mid, high, level-1)
    

def roc_finite_sample_band(n_samples, mc_samples=10000):
    """Using MC, find the 95% containment band for ROC curve for a gaussian distribution."""
    invcdf_arr = []
    for _ in range(mc_samples):
        x_sample = np.random.normal(size=n_samples)
        p_sample = (special.erf(np.abs(x_sample)/np.sqrt(2)) - special.erf(-np.abs(x_sample)/np.sqrt(2))) / 2
        invcdf_arr.append(np.sort(p_sample))
    invcdf_arr = np.array(invcdf_arr)
    invcdf_upper = np.quantile(invcdf_arr, 0.975, axis=0)
    invcdf_lower = np.quantile(invcdf_arr, 0.025, axis=0)
    return invcdf_lower, invcdf_upper