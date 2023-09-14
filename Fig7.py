# Fig7.py
# Generate my two rows of figure 7 of the final manuscript.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy import stats
from sklearn.decomposition import PCA
from tqdm import tqdm

import hmmsupport
from hmmsupport import get_raster, figure, Model, all_experiments, load_metrics


source = "org_and_slice"
experiments = all_experiments(source)
groups = {"L": "Organoid", "M": "Slice", "Pr": "Primary"}
groups_with_all = {**groups, "": "Overall"}
exp_to_model = {e: next(k for k in groups if e.startswith(k)) for e in experiments}

plt.ion()
hmmsupport.figdir("paper")

bin_size_ms = 30
n_stateses = np.arange(10, 51)

print("Loading real and surrogate rasters and doing PCA on HMMs.")
with tqdm(total=2 * len(experiments) * (1 + len(n_stateses))) as pbar:
    rasters_real, rasters_rsm, backbone, nonrigid = {}, {}, {}, {}
    _rs = dict(real=rasters_real, rsm=rasters_rsm)
    for exp in experiments:
        metrics = load_metrics(exp, only_include=["scaf_units", "non_scaf_units"])
        backbone[exp] = np.int32(metrics["scaf_units"].ravel()) - 1
        nonrigid[exp] = np.int32(metrics["non_scaf_units"].ravel()) - 1
        for surr in ["real", "rsm"]:
            _rs[surr][exp] = get_raster(source, exp, bin_size_ms, surr), []
            pbar.update()
            for n in n_stateses:
                m = Model(source, exp, bin_size_ms, n, surr)
                m.pca = PCA().fit(np.exp(m._hmm.observations.log_lambdas))
                _rs[surr][exp][1].append(m)
                pbar.update()

for k, (r_real, _) in rasters_real.items():
    nunits = r_real._raster.shape[1]
    meanfr = r_real._raster.mean() / r_real.bin_size_ms * 1000
    nbursts = len(r_real.find_bursts())
    print(
        f"{k} has {nunits} units firing at {meanfr:0.2f} " f"Hz with {nbursts} bursts"
    )


# Vectors of the first ten explained variance ratios for all of the
# experiments.
pve_real, pve_rsm = [
    {
        e: np.array([m.pca.explained_variance_ratio_[:10] for m in ms])
        for e, (_, ms) in rs.items()
    }
    for rs in [rasters_real, rasters_rsm]
]


def components_required(exp, thresh, rsm=False):
    enough = np.cumsum((pve_rsm if rsm else pve_real)[exp], axis=1) > thresh
    return [
        np.argmax(enough[i, :]) + 1 if np.any(enough[i, :]) else enough.shape[1] + 1
        for i in range(enough.shape[0])
    ]


def pev_vs_thresholds(experiments, xs, rsm=False):
    return np.array(
        [
            np.hstack(
                [components_required(exp, thresh=x, rsm=rsm) for exp in experiments]
            )
            for x in xs
        ]
    )


def plot_pev(ax, color, experiments, label, rsm=False):
    xs = np.linspace(0.705, 1)
    pev = pev_vs_thresholds(experiments, xs, rsm=rsm)
    ys = pev.mean(axis=1)
    ystd = pev.std(axis=1)
    ax.fill_between(xs, ys - ystd, ys + ystd, alpha=0.2, color=color, label=label)
    ax.plot(xs, ys, color=color)


def poisson_test(data, mean=None):
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


def all_the_scores(exp, only_burst=False):
    """
    Gather all the consistency scores and observation counts across all states of
    all models. Don't differentiate between states or models, yielding a 2D array
    with some large number of rows and one column per unit.
    """
    scores_nobs = [
        unit_consistency(model, rasters_real[exp][0], only_burst)
        for model in rasters_real[exp][1]
    ]
    scores = np.vstack([s for s, _ in scores_nobs])
    nobs = np.hstack([n for _, n in scores_nobs])
    return scores, nobs


def mean_consistency(scores_nobs, include_nan=True, weight=True):
    """
    Combine p-values for the Poisson null hypothesis all states of all models,
    reducing an array of size (M,N) to (N,) so that there is just one score per
    unit. Returns a vector of shape (n_units,) giving the fraction of all states
    across all models where the unit is too consistent to be Poisson.

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
    return np.ma.average(scores < 0.01, axis=0, weights=weights)


def unit_consistency(model, raster, only_burst=False):
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


consistencies = {}
for exp in tqdm(experiments):
    consistencies[exp] = mean_consistency(
        all_the_scores(exp, True),
        include_nan=True,
    )

df = []
for exp in experiments:
    for unitgroup in [backbone, nonrigid]:
        label = 1 if unitgroup is backbone else 0
        df.extend(
            dict(
                experiment=exp,
                model=groups[exp_to_model[exp]],
                consistency=c,
                label=label,
                backbone="Backbone" if label else "Non-Rigid",
            )
            for c in consistencies[exp][unitgroup[exp]]
        )
df = pd.DataFrame(df)

# %%


with figure("Fig7", figsize=(8.5, 3.0)) as f:
    G, H = f.subplots(1, 2, gridspec_kw=dict(width_ratios=[2, 3]))

    # Subfigure G: dimensionality as a function of PC inclusion threshold.
    for i, prefix in enumerate(groups):
        expsub = [e for e in experiments if e.startswith(prefix)]
        plot_pev(G, f"C{i}", expsub, groups[prefix])
    plot_pev(G, "red", experiments, "Surrogate", rsm=True)
    G.legend(loc="upper left")
    G.set_xlabel("Explained Variance Threshold")
    G.set_ylabel("Dimensions Required")
    G.set_ylim(1, 6)
    G.set_xlim(0.7, 1)
    G.xaxis.set_major_formatter(PercentFormatter(1, 0))

    # Subfigure H: split violins of consistency by backbone/non-rigid.
    sns.violinplot(
        bw=0.1,
        data=df,
        ax=H,
        split=True,
        x="model",
        y="consistency",
        hue="backbone",
        inner=None,
        cut=0,
        scale="area",
    )
    H.set_ylabel("Fraction of States with Non-Poisson Firing")
    H.set_xlabel(None)
    H.legend(loc="upper right")
