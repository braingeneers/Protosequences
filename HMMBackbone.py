# HMMBackbone.py
#
# Attempt to identify the backbone units within each recording based on how consistent
# they are within different HMM states. Also check the consistency of this measurement
# across the different models trained for a given experiment.
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from scipy import stats

import hmmsupport
from hmmsupport import get_raster, all_experiments, Model, figure

plt.ion()
hmmsupport.figdir("pca")

source = "org_and_slice"
bin_size_ms = 30
n_stateses = np.arange(10, 21)

experiments = {v.split("_")[0]: v for v in all_experiments(source)}


rasters, bad_rasters = {}, {}
with tqdm(desc="Loading rasters", total=2 * len(experiments), delay=0.1) as pbar:
    for k, exp in experiments.items():
        for rs, surr in [(rasters, "real"), (bad_rasters, "rsm")]:
            rs[k] = get_raster(source, exp, bin_size_ms, surr)
            pbar.update()

models, bad_models = {}, {}
with tqdm(
    desc="Loading models", total=2 * len(experiments) * len(n_stateses), delay=0.1
) as pbar:
    for k, exp in experiments.items():
        for ms, surr in [(models, "real"), (bad_models, "rsm")]:
            ms[k] = []
            for n in n_stateses:
                ms[k].append(
                    Model(source, exp, bin_size_ms, n, surr, recompute_ok=False)
                )
                pbar.update()

metrics = {}
with tqdm(desc="Loading SRMs", total=len(experiments), delay=0.1) as pbar:
    for k, exp in experiments.items():
        metrics[k] = hmmsupport.load_metrics(exp)
        pbar.update()

# %%


def poisson_test_students_t(data, axis=0):
    """
    Test the null hypothesis that the given data has no less variance than a
    Poisson distribution with the same mean, using a t-test to check whether the
    second moment is consistent with the first.
    """
    if len(data) == 0:
        return np.full_like(data.sum(axis), np.nan)

    # Expected value of y² when y ~ Poisson(λ) is λ² + λ.
    λ = data.mean(axis=axis, keepdims=True)
    return stats.ttest_1samp(
        data**2, λ**2 + λ, axis=axis, alternative="less"
    ).pvalue


def poisson_test_monte_carlo(data, axis=0):
    """
    Test the null hypothesis that the given data has no less variance than a
    Poisson distribution with the same mean, using an explicit Monte Carlo test
    to check whether the sample variance is smaller than Poisson.
    """
    if len(data) == 0:
        return np.full_like(data.sum(axis), np.nan)

    return stats.monte_carlo_test(
        data,
        stats.poisson(data.mean()).rvs,
        statistic=np.var,
        alternative="less",
        axis=axis,
        n_resamples=1000,
    ).pvalue


def unit_consistency(model, raster, poisson_test):
    """
    Compute the consistency of each unit within each state of the given model,
    by grouping the time bins of the raster according to the state assigned by
    the model and computing the COV of the firing rate of each unit across the
    realizations of each state.

    Returns a matrix of shape (n_states, n_units) with the consistency of each
    unit in each state.
    """
    K = model.n_states
    h = model.states(raster)
    ret = [poisson_test(raster._raster[h == state, :]) for state in range(K)]
    return np.array(ret)


def overall_consistency(exp, poisson_test):
    """
    Compute the consistency of each unit across all states of all models trained
    for the given experiment. Returns a vector of shape (n_units,) with the
    fraction of all states across all models where the Poisson null hypothesis
    failed to be rejected for that unit.
    """
    # Compare using less-than because the chi-squared test often returns NaN in
    # indeterminate cases. This way, those are marked NOT rejected.
    reject = (
        np.vstack(
            [
                unit_consistency(model, rasters[exp], poisson_test)
                for model in models[exp]
            ]
        )
        < 0.01
    )
    # Then take the complement so the score represents positive consistency.
    return 1 - reject.mean(0)


consistencies = {}
for k in tqdm(experiments, desc="Computing consistency"):
    consistencies[k] = overall_consistency(k, poisson_test_students_t)


# %%

def convertix(exp, key):
    """
    Convert a weird 2D matrix with potentially floating-point "indices" into a
    proper zero-based index array.
    """
    return np.int64(metrics[exp][key]).ravel() - 1

import pandas as pd
import seaborn as sns

groups = dict(L="Organoid", M="Mouse", Pr="Primary")

with figure("Unit Consistency", save_exts=[]) as f:
    rows = []
    for prefix in groups:
        for exp in [e for e in experiments if e.startswith(prefix)]:
            for key in ["scaf_units", "non_scaf_units"]:
                rows.extend([dict(
                    consistency=c,
                    backbone="Backbone" if key == "scaf_units" else "Non-Rigid",
                    model=prefix,
                ) for c in consistencies[exp][convertix(exp, key)]])

    ax = f.gca()
    sns.violinplot(data=pd.DataFrame(rows), ax=ax, split=True, x="model",
                   y="consistency", hue="backbone", inner="quartile",
                   saturation=0.7, cut=0, scale="count")
    ax.set_xticks([0, 1, 2], [groups[g] for g in groups])
    ax.set_ylabel("Fraction of States Indistinguishable from Poisson")
    ax.set_xlabel("")
    # ax.set_ylim(0.4, 1.0)
    ax.legend()
