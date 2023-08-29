# HMMBackbone.py
#
# Attempt to identify the backbone units within each recording based on how consistent
# they are within different HMM states. Also check the consistency of this measurement
# across the different models trained for a given experiment.
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm

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


def poisson_test_students_t(data, mean=None):
    """
    Test the null hypothesis that the given data of shape (N,K) has no less
    variance than a K-dimensional multivariate Poisson distribution with the same
    mean, using a t-test to check whether the second moment is consistent with the
    first.

    Provide the mean (derived from some other source) if you want to take the
    p-value seriously! It has degrees-of-freedom problems when the mean is
    estimated from the data, and the variance of the test statistic ends up way
    too small.
    """
    if len(data) == 0:
        return np.full(data.sum(0).shape, np.nan)

    # Expected value of y² when y ~ Poisson(λ) is λ² + λ.
    λ = data.mean(0, keepdims=True) if mean is None else mean
    return stats.ttest_1samp(
        data**2, λ**2 + λ, axis=0, alternative="less"
    ).statistic


def poisson_test_chi_square(data, mean=None):
    """
    Test the null hypothesis that the given data of shape (N,K) has no less
    variance than a K-dimensional multivariate Poisson distribution with the same
    mean, using a chi-square test to check whether the second moment is consistent
    with the first.

    This is even uniform under the null hypothesis when the mean is generated
    from the data. :D
    """
    if len(data) == 0:
        return np.full(data.sum(0).shape, np.nan)

    # Dividing by the expected variance, which for Poisson is the mean.
    mean = data.mean(0) if mean is None else mean
    statistic = (data.shape[0] - 1) * data.var(0) / mean
    # Use the CDF to get the p-value: how likely a χ² sample is to be smaller
    # than the observed test statistic.
    return stats.chi2(data.shape[0] - 1).cdf(statistic)


def poisson_test_monte_carlo(data, mean=None):
    """
    Test the null hypothesis that the given data of shape (N,K) has no less
    variance than a K-dimensional multivariate Poisson distribution with the same
    mean, using an explicit Monte Carlo test generating N samples from the same
    distribution M times to check whether the sample variance is smaller than
    Poisson.

    All the random variables for each of the K input dimensions are generated in
    one batch, so this function might use a lot of memory. It's also slow as heck,
    and unavoidably so, so try to use the χ² test instead.
    """
    if len(data) == 0:
        return np.full(data.sum(0).shape, np.nan)

    λ = data.mean(0) if mean is None else mean
    return np.array(
        [
            stats.monte_carlo_test(
                row,
                stats.poisson(li).rvs,
                np.var,
                alternative="less",
                n_resamples=1000,
            ).pvalue
            for li, row in zip(λ, data.T)
        ]
    )


def unit_consistency(model, raster: hmmsupport.Raster, poisson_test, only_burst=False):
    """
    Compute the consistency of each unit within each state of the given model,
    by grouping the time bins of the raster according to the state assigned by
    the model and computing the COV of the firing rate of each unit across the
    realizations of each state.

    Returns a matrix of shape (n_states, n_units) with the consistency of each
    unit in each state, and an array with shape (n_states,) with the number of
    observations of each state.
    """
    K = model.n_states
    h = model.states(raster)

    if only_burst:
        peaks, bounds = raster.find_burst_edges()
        edges = np.int64(np.round((peaks[:, None] + bounds) / raster.bin_size_ms))
        ok = np.zeros_like(h, dtype=bool)
        for start, end in edges:
            ok[start:end] = True
        h[~ok] = -1

    rsubs = [raster._raster[h == state, :] for state in range(K)]
    ret = [poisson_test(rsub) for rsub in rsubs]
    return np.array(ret), np.array([rsub.shape[0] for rsub in rsubs])


@functools.lru_cache(maxsize=50)
def all_the_scores(exp, poisson_test, only_burst=False):
    """
    Gather all the consistency scores and observation counts across all states of
    all models. Don't differentiate between states or models, yielding a 2D array
    with some large number of rows and one column per unit.
    """
    scores_nobs = [
        unit_consistency(model, rasters[exp], poisson_test, only_burst)
        for model in models[exp]
    ]
    scores = np.vstack([s for s, _ in scores_nobs])
    nobs = np.hstack([n for _, n in scores_nobs])
    return scores, nobs


def mean_consistency(scores_nobs, include_nan=True):
    """
    Combine consistency scores across all states of all models, reducing an array
    of size (M,N) to (N,) so that there is just one score per unit. Returns a
    vector of shape (n_units,) giving the fraction of all states across all models
    where the Poisson null hypothesis failed to be rejected for that unit.

    If `include_nan` is True, then (state, unit) combinations with zero events
    are included in the calculation and considered indistinguishable from
    Poisson. Otherwise, they are excluded from the calculation entirely.
    """
    scores, _ = scores_nobs
    # Compare in the correct sense so that NaNs are treated as "not
    # consistent", i.e. potentially Poisson, then invert.
    return 1 - (scores < 0.01).mean(0, where=include_nan or ~np.isnan(scores))


def p_consistency(scores_nobs, include_nan=True):
    """
    Combine the consistency scores using methods for combining the p-value
    instead of just averaging like mean_consistency().
    """
    scores, nobs = scores_nobs
    if include_nan:
        scores = np.where(np.isnan(scores), 0.5, scores)

    def _combine(col, weights):
        mask = ~np.isnan(col)
        return stats.combine_pvalues(
            col[mask], method="stouffer", weights=weights[mask]
        ).pvalue

    return np.apply_along_axis(_combine, 0, scores, nobs)


consistencies = {}
for k in tqdm(experiments, desc="Computing consistency"):
    consistencies[k] = p_consistency(
        all_the_scores(k, poisson_test_chi_square, False), False
    )

# %%


def convertix(exp, key):
    """
    Convert a weird 2D matrix with potentially floating-point "indices" into a
    proper zero-based index array.
    """
    return np.int64(metrics[exp][key]).ravel() - 1


groups = dict(L="Organoid", M="Mouse", Pr="Primary")

with figure("Unit Consistency") as f:
    rows = []
    for prefix in groups:
        for exp in [e for e in experiments if e.startswith(prefix)]:
            for key in ["scaf_units", "non_scaf_units"]:
                rows.extend(
                    dict(
                        consistency=c,
                        backbone="Backbone" if key == "scaf_units" else "Non-Rigid",
                        model=prefix,
                    )
                    for c in consistencies[exp][convertix(exp, key)]
                )
    data = pd.DataFrame(rows)

    ax = f.gca()
    sns.violinplot(
        data=data,
        ax=ax,
        split=True,
        x="model",
        y="consistency",
        hue="backbone",
        inner="quartile",
        saturation=0.7,
        cut=0,
        scale="count",
    )
    ax.set_xticks([0, 1, 2], [groups[g] for g in groups])
    ax.set_ylabel("Fraction of States Indistinguishable from Poisson")
    ax.set_xlabel("")
    ax.legend()


def subgroup_ks(model):
    c_bb = data.loc[
        (data.model == model) & (data.backbone == "Backbone"), "consistency"
    ]
    c_nr = data.loc[
        (data.model == model) & (data.backbone == "Non-Rigid"), "consistency"
    ]
    return stats.ks_2samp(c_bb, c_nr)


for group in groups:
    print(f"{groups[group]}: {subgroup_ks(group)}")
