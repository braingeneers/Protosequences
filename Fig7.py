# Fig7.py
# Generate my two rows of figure 7 of the final manuscript.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy import stats
from sklearn.decomposition import PCA
from tqdm import tqdm

import hmmsupport
from hmmsupport import Model, all_experiments, figure, get_raster, load_metrics

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


def components_required(exp, thresh, rsm=False, which_models=None):
    scores = (pve_rsm if rsm else pve_real)[exp]
    if which_models is not None:
        K_min, K_max = which_models
        scores = scores[(n_stateses >= K_min) & (n_stateses <= K_max), :]
    enough = np.cumsum(scores, axis=1) > thresh
    return [
        np.argmax(enough[i, :]) + 1 if np.any(enough[i, :]) else enough.shape[1] + 1
        for i in range(enough.shape[0])
    ]


def dimensions_required(experiments, xs, rsm=False, which_models=None):
    return np.array(
        [
            np.hstack(
                [
                    components_required(exp, x, rsm=rsm, which_models=which_models)
                    for exp in experiments
                ]
            )
            for x in xs
        ]
    )


def plot_dimensions_required(
    ax, color, experiments, label, rsm=False, which_models=None
):
    xs = np.linspace(0.705, 1)
    dreq = dimensions_required(experiments, xs, rsm, which_models)
    ys = np.mean(dreq, axis=1)
    ystd = np.std(dreq, axis=1)
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


def all_the_scores(exp, only_burst=False, rsm=False):
    """
    Gather all the consistency scores and observation counts across all states of
    all models. Don't differentiate between states or models, yielding a 2D array
    with some large number of rows and one column per unit.
    """
    r, models = (rasters_rsm if rsm else rasters_real)[exp]
    scores_nobs = [unit_consistency(model, r, only_burst) for model in models]
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


consistencies = {
    e: mean_consistency(all_the_scores(e, True, rsm=False)) for e in tqdm(experiments)
}

df = pd.DataFrame(
    [
        dict(
            experiment=exp,
            model=groups[exp_to_model[exp]],
            consistency=c,
            label=label,
            backbone="Backbone" if label else "Non-Rigid",
        )
        for exp in experiments
        for label, unitgroup in enumerate([nonrigid, backbone])
        for c in consistencies[exp][unitgroup[exp]]
    ]
)

# %%


def fraction_above_xs(xs, data, backbone=None, model=None):
    """
    Compute the fraction of data points above each threshold in `xs`.
    """
    if backbone is not None:
        data = data[data.label == int(backbone)]
    if model is not None:
        data = data[data.model == model]
    return np.array(
        [
            [(dsub.consistency >= x).mean() for x in xs]
            for _, dsub in data.groupby("experiment")
        ]
    )


with figure("Fig7", figsize=(8.5, 3.0)) as f:
    G, H = f.subplots(1, 2, gridspec_kw=dict(width_ratios=[2, 3]))

    # Subfigure G: dimensionality as a function of PC inclusion threshold.
    which_models = 10, 30
    for i, prefix in enumerate(groups):
        expsub = [e for e in experiments if e.startswith(prefix)]
        plot_dimensions_required(
            G, f"C{i}", expsub, groups[prefix], which_models=which_models
        )
    plot_dimensions_required(
        G, "red", experiments, "Surrogate", rsm=True, which_models=which_models
    )
    G.legend(loc="upper left")
    G.set_xlabel("Explained Variance Threshold")
    G.set_ylabel("Dimensions Required")
    G.set_ylim(1, 6)
    G.set_xlim(0.7, 1)
    G.xaxis.set_major_formatter(PercentFormatter(1, 0))

    # Subfigure H: split violins of consistency by backbone/non-rigid.
    xs = np.linspace(0.5, 1, 100)
    for model in ["Organoid", "Slice", "Primary"]:
        y_bb = fraction_above_xs(xs, df, True, model)
        y_nr = fraction_above_xs(xs, df, False, model)
        ys = y_bb - y_nr
        H.semilogy(xs, ys.mean(0), label=model)
        H.fill_between(
            xs,
            ys.mean(0) - ys.std(0),
            ys.mean(0) + ys.std(0),
            alpha=0.5,
            color=H.get_lines()[-1].get_color(),
        )
    H.set_ylim(1e-3, 1)
    H.set_xlim(0.5, 1)
    H.legend()
    H.set_xlabel("Non-Poisson Threshold")
    H.set_ylabel("Fraction of Non-Poisson Units")


# %%

which_models = 10, 50
dimensions = {}
xs = np.linspace(0, 1, num=100)[1:]
for i, prefix in enumerate(groups):
    expsub = [e for e in experiments if e.startswith(prefix)]
    dimensions[prefix] = dimensions_required(expsub, xs, which_models=which_models)
dimensions["*"] = dimensions_required(
    experiments, xs, which_models=which_models, rsm=True
)

for a, b in [("L", "M"), ("M", "Pr"), ("L", "Pr"), ("M", "*"), ("L", "*"), ("Pr", "*")]:
    scores = stats.mannwhitneyu(dimensions[a], dimensions[b], axis=1).pvalue
    print(a, b, stats.gmean(scores[~np.isnan(scores)]))


# %%

with figure("Supplement to Fig7") as f:
    ax = f.gca()
    which_models = 10, 50
    for i, prefix in enumerate(groups):
        expsub = [e for e in experiments if e.startswith(prefix)]
        plot_dimensions_required(
            ax, f"C{i}", expsub, groups[prefix], which_models=which_models
        )
    plot_dimensions_required(
        ax, "red", experiments, "Surrogate", rsm=True, which_models=which_models
    )
    ax.legend(loc="upper left")
    ax.set_xlabel("Explained Variance Threshold")
    ax.set_ylabel("Dimensions Required")
    ax.set_ylim(1, 6)
    ax.set_xlim(0.7, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1, 0))
