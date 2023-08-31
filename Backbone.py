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
from sklearn.metrics import roc_curve, roc_auc_score

import hmmsupport
from hmmsupport import get_raster, all_experiments, Model, figure


plt.ion()
hmmsupport.figdir("pca")

source = "org_and_slice"
bin_size_ms = 30
n_stateses = np.arange(10, 21)

experiments = {v.split("_")[0]: v for v in all_experiments(source)}
groups = {"L": "Organoid", "M": "Mouse", "Pr": "Primary", "": "All"}


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


def mean_consistency(scores_nobs, include_nan=True, weight=True):
    """
    Combine consistency scores across all states of all models, reducing an array
    of size (M,N) to (N,) so that there is just one score per unit. Returns a
    vector of shape (n_units,) giving the fraction of all states across all models
    where the Poisson null hypothesis failed to be rejected for that unit.

    If `include_nan` is True, then (state, unit) combinations with zero events
    are included in the calculation and considered indistinguishable from
    Poisson. Otherwise, they are excluded from the calculation entirely.
    """
    scores, nobs = scores_nobs
    if not include_nan:
        scores = np.ma.array(scores, mask=np.isnan(scores))
    # Compare in the correct sense so that NaNs are treated as "not
    # consistent", i.e. potentially Poisson, then invert.
    weights = nobs if weight else None
    ret = 1 - np.ma.average(scores < 0.01, axis=0, weights=weights)
    # We shouldn't see any NaNs here, but better safe than sorry. ;)
    return np.where(np.isnan(ret), 0.5, ret)


def p_consistency(scores_nobs, include_nan=True, method="stouffer"):
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
            col[mask], method=method, weights=weights[mask]
        ).pvalue

    ret = np.apply_along_axis(_combine, 0, scores, nobs)
    return np.where(np.isnan(ret), 0.5, ret)


def convertix(exp, key):
    """
    Convert a weird 2D matrix with potentially floating-point "indices" into a
    proper zero-based index array.
    """
    return np.int64(metrics[exp][key]).ravel() - 1


def consistency_data(method, only_burst, include_nan, pbar=None):
    """
    Use the given consistency score combining method to create a dataframe of
    consistency scores which can be used for further processing.

    Displays progress on a progress bar, which can be provided as an argument
    (pass pbar=False to disable).
    """
    # If pbar is None, create a new progress bar, but if it's False, don't show a
    # progress bar at all. Otherwise, use the one passed in. Use the context
    # manager to ensure the progress bar is closed when we're done.
    if not pbar:
        with tqdm(
            total=len(experiments), desc="Computing consistency", disable=pbar == False
        ) as pbar:
            return consistency_data(method, only_burst, include_nan, pbar)

    consistencies = {}
    for k in experiments:
        consistencies[k] = method(
            all_the_scores(k, poisson_test_chi_square, only_burst),
            include_nan=include_nan,
        )
        pbar.update()

    rows = []
    for prefix in groups:
        for exp in [e for e in experiments if e.startswith(prefix)]:
            for key in ["scaf_units", "non_scaf_units"]:
                rows.extend(
                    dict(
                        consistency=c,
                        label=1 if key == "scaf_units" else 0,
                        backbone="Backbone" if key == "scaf_units" else "Non-Rigid",
                        model=prefix,
                    )
                    for c in consistencies[exp][convertix(exp, key)]
                )
    return pd.DataFrame(rows)


# %%


conditions = [
    (combining, include_nan, only_burst)
    for combining in [
        ("p-Value", "$p$-value for Non-Poisson Firing", p_consistency),
        ("Mean", "Fraction of States with Non-Poisson Firing", mean_consistency),
    ]
    for include_nan in [False, True]
    for only_burst in [False, True]
]


with tqdm(
    total=len(conditions) * len(experiments), desc="Computing consistency"
) as pbar:
    for (kind, label, score_combiner), include_nan, only_burst in conditions:
        data = consistency_data(score_combiner, only_burst, include_nan, pbar)
        nanlabel = "With NaN" if include_nan else "Without NaN"
        burstlabel = "Bursts Only" if only_burst else "All Bins"
        with figure(f"Unit {kind} Consistency, {nanlabel}, {burstlabel}") as f:
            ax = f.gca()
            sns.violinplot(
                bw=0.1,
                data=data,
                ax=ax,
                split=True,
                x="model",
                y="consistency",
                hue="backbone",
                inner="quartile",
                cut=0,
                scale="count",
            )
            ax.set_xticks([0, 1, 2, 3], [groups[g] for g in groups])
            ax.set_ylabel(label)
            ax.set_xlabel("")
            ax.legend()

        for model in groups:
            c_bb = data.loc[
                (data.model == model) & (data.backbone == "Backbone"), "consistency"
            ]
            c_nr = data.loc[
                (data.model == model) & (data.backbone == "Non-Rigid"), "consistency"
            ]
            stat = stats.ks_2samp(c_bb, c_nr)
            if stat.pvalue > 0.001:
                print(f"{groups[model]}: {stats.ks_2samp(c_bb, c_nr)}")


# %%


def auc_pval(auc, labels):
    """
    Calculate a p-value for the given classifier AUC.
    """
    ep = (labels != 0).sum()
    en = len(labels) - ep

    # The Mann-Whitney statistic is normally distributed under the null
    # hypothesis, with the following parameters.
    μ = ep * en / 2
    σ = np.sqrt(ep * en * (1 + ep + en) / 12)

    # The Mann-Whitney statistic is equal to ep*en*(1 - AUC), so use the SF
    # directly to calculate the p-value of an AUC this large.
    return stats.norm(μ, σ).cdf((1 - auc) * ep * en)


rows = []
for (kind, label, combiner), include_nan, only_burst in conditions:
    data = consistency_data(combiner, only_burst, include_nan, False)
    nanlabel = "With NaN" if include_nan else "Without NaN"
    burstlabel = "Bursts Only" if only_burst else "All Bins"
    with figure(f"{kind} Consistency ROC, {nanlabel}, {burstlabel}"):
        for i, (group, subdata) in enumerate(data.groupby("model")):
            auc = roc_auc_score(subdata.label, 1 - subdata.consistency)
            fpr, tpr, thresh = roc_curve(subdata.label, 1 - subdata.consistency)
            # Calculate the best accuracy as well as the null accuracy.
            baserate = subdata.label.mean()
            null_acc = max(baserate, 1 - baserate)
            accuracy = tpr * baserate + (1 - fpr) * (1 - baserate)
            rows.append(
                dict(
                    combiner=kind,
                    include_nan=include_nan,
                    nan=nanlabel,
                    only_burst=only_burst,
                    burst=burstlabel,
                    prefix=group,
                    group=groups[group],
                    auc=auc,
                    acc=accuracy.max(),
                    null_acc=null_acc,
                    pvalue=auc_pval(auc, subdata.label),
                )
            )
            plt.plot(fpr, tpr, label=groups[group])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

aucs = pd.DataFrame(rows)


# %%

for key in ["nan", "burst", "combiner"]:
    with figure(f"AUC by Model and {key}") as f:
        sns.violinplot(
            aucs,
            x="group",
            y="auc",
            split=True,
            hue=key,
            inner="quartile",
        )
