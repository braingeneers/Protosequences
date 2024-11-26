# Fig7.py
# Generate my two rows of figure 7 of the final manuscript.
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed, parallel_backend
from matplotlib.ticker import PercentFormatter
from scipy import stats
from sklearn.decomposition import PCA
from tqdm import tqdm

from hmmsupport import (
    ALL_EXPERIMENTS,
    DATA_SOURCE,
    EXPERIMENT_GROUP,
    GROUP_EXPERIMENTS,
    GROUP_NAME,
    SHORT_NAME,
    Model,
    Raster,
    figure,
    get_raster,
    load_metrics,
    separability,
)

EXP_RMS = (
    {exp: 5.0 for exp in GROUP_EXPERIMENTS["HO"]}
    | {exp: 3.0 for exp in GROUP_EXPERIMENTS["MO"]}
    | {exp: 6.0 for exp in GROUP_EXPERIMENTS["MS"]}
    | {exp: 3.0 for exp in GROUP_EXPERIMENTS["Pr"]}
    | {"MO1_t_spk_mat_sorted": 2.5, "MO2_t_spk_mat_sorted": 2.0}
)

plt.ion()

bin_size_ms = 30
n_stateses = np.arange(10, 31)


backbone, nonrigid, unit_orders = {}, {}, {}
which_metrics = ["scaf_units", "non_scaf_units", "mean_rate_ordering"]
print("Loading metrics files.")
for exp in tqdm(ALL_EXPERIMENTS):
    metrics = load_metrics(exp, only_include=which_metrics)
    backbone[exp] = np.int32(metrics["scaf_units"].ravel()) - 1
    nonrigid[exp] = np.int32(metrics["non_scaf_units"].ravel()) - 1
    unit_orders[exp] = np.int32(metrics["mean_rate_ordering"].flatten()) - 1


print("Loading real and surrogate rasters and doing PCA on HMMs.")
with tqdm(total=2 * len(ALL_EXPERIMENTS) * (1 + len(n_stateses))) as pbar:
    _rs: dict[str, dict[str, tuple[Raster, list[Model]]]] = dict(real={}, rsm={})
    rasters_real, rasters_rsm = _rs["real"], _rs["rsm"]
    for exp in ALL_EXPERIMENTS:
        for surr in ["real", "rsm"]:
            r, ms = _rs[surr][exp] = get_raster(DATA_SOURCE, exp, bin_size_ms, surr), []
            r.burst_rms = EXP_RMS[exp]
            r.unit_order = unit_orders[exp]
            pbar.update()
            for n in n_stateses:
                m = Model(DATA_SOURCE, exp, bin_size_ms, n, surr)
                m.compute_consistency(r)
                m.pca = PCA().fit(np.exp(m._hmm.observations.log_lambdas))
                ms.append(m)
                pbar.update()

for exp, (r_real, _) in rasters_real.items():
    nunits = r_real._raster.shape[1]
    meanfr = r_real._raster.mean() / r_real.bin_size_ms * 1000
    nbursts = len(r_real.find_bursts())
    print(
        f"{exp} has {nunits} units firing at {meanfr:0.2f} " f"Hz with {nbursts} bursts"
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
    e: mean_consistency(all_the_scores(e, True, rsm=False))
    for e in tqdm(ALL_EXPERIMENTS)
}

consistency_df = pd.DataFrame(
    [
        dict(
            experiment=exp,
            model=GROUP_NAME[EXPERIMENT_GROUP[exp]],
            consistency=c,
            label=label,
            backbone="Backbone" if label else "Non-Rigid",
        )
        for exp in ALL_EXPERIMENTS
        for label, unitgroup in enumerate([nonrigid, backbone])
        for c in consistencies[exp][unitgroup[exp]]
    ]
)


with parallel_backend("loky", n_jobs=12):
    sep_on_states = {
        exp: [separability(m.consistency.T, len(nonrigid[exp])) for m in ms]
        for exp, (_, ms) in rasters_real.items()
    }

    sep_on_states_rsm = {
        exp: [separability(m.consistency.T, len(nonrigid[exp])) for m in ms]
        for exp, (_, ms) in rasters_rsm.items()
    }

    sep_on_fr = {
        exp: separability(r.rates("Hz").reshape((-1, 1)), len(nonrigid[exp]))
        for exp, (r, _) in rasters_real.items()
    }

separability_df = pd.DataFrame(
    [
        dict(
            sample_type=EXPERIMENT_GROUP[exp],
            sample_id=SHORT_NAME[exp],
            K=n_stateses[i],
            sep_on_states=on_states,
            sep_on_states_rsm=sep_on_states_rsm[exp][i],
            sep_on_fr=sep_on_fr[exp],
        )
        for exp, on_stateses in sep_on_states.items()
        for i, on_states in enumerate(on_stateses)
    ]
)
separability_df.to_csv("separability.csv", index=False)
separability_df["model"] = separability_df.sample_type.map(lambda g: GROUP_NAME[g])


def where_bb(exp, A, B):
    ret = A.copy()
    ret[:, backbone[exp]] = B[:, backbone[exp]]
    return ret


def _el(exp, r_real, r_rsm, models):
    return [
        dict(
            sample_type=EXPERIMENT_GROUP[exp],
            sample_id=SHORT_NAME[exp],
            K=n_stateses[i],
            ll_real=model._hmm.log_likelihood(r_real),
            ll_rsm=model._hmm.log_likelihood(r_rsm),
            ll_backbone=model._hmm.log_likelihood(where_bb(exp, r_rsm, r_real)),
            ll_nonrigid=model._hmm.log_likelihood(where_bb(exp, r_real, r_rsm)),
        )
        for i, model in enumerate(models)
    ]


ll_df = pd.DataFrame(
    chain.from_iterable(
        Parallel(n_jobs=12)(
            delayed(_el)(exp, r._raster, rasters_rsm[exp][0]._raster, models)
            for exp, (r, models) in rasters_real.items()
        ),
    )
)
ll_df.to_csv("ll.csv", index=False)


dreq_df = pd.DataFrame(
    [
        dict(
            sample_type=EXPERIMENT_GROUP[exp],
            sample_id=SHORT_NAME[exp],
            theta=theta,
            K=K,
            dims=d,
            dims_rsm=dr,
            dims_norm=(d - dr) / (d + dr),
        )
        for exp in ALL_EXPERIMENTS
        for theta in np.linspace(0.5, 1, num=51)
        for K, d, dr in zip(
            n_stateses,
            components_required(exp, theta),
            components_required(exp, theta, rsm=True),
        )
    ]
)
dreq_df.to_csv("dimensions.csv", index=False)

# %%
# Figure 7E: distribution of HMM dimensionality (to explain 75% of variance)
# compared between all the models.

dsub = dreq_df[dreq_df.theta == 0.75]
dsub.to_csv("dimensions_0.75.csv", index=False)

with figure("Fig7E", figsize=(4.0, 3.0), save_exts=["png", "svg"]) as f:
    axes = f.subplots(4, 1)
    for (i, group), ax in zip(enumerate(GROUP_NAME), axes):
        sns.histplot(
            data=dsub[dsub.sample_type == group],
            x="dims",
            stat="count",
            discrete=True,
            color=f"C{i}",
            ax=ax,
        )
        ax.set_xlim(0.35, 7.65)
        ax.set_ylabel(group)
        ax.set_xticks([])
        ax.set_xlabel("")
    ax.set_xlabel("Dimensions to Explain 75\\% of Variance")
    ax.set_xticks(np.arange(1, 8))

with figure("Fig7E Alternate") as f:
    ax = f.gca()
    sns.violinplot(
        data=dsub, x="sample_type", y="dims", bw=0.6, cut=0, inner=None, ax=ax
    )
    ax.set_ylim(*ax.get_ylim())
    sections = [
        dsub[dsub.sample_type == label.get_text()] for label in ax.get_xticklabels()
    ]
    means = [section["dims"].mean() for section in sections]
    stds = [section["dims"].std() for section in sections]
    ax.errorbar(ax.get_xticks(), means, yerr=stds, fmt="ko")
    ax.set_xlabel("")
    ax.set_ylabel("Dimensions to Explain 75\\% of Variance")



# %%
# S25: supplemement to figure 7E showing the dimensions required and the
# significance as a function of threshold θ.
#
# Make sure to run lmem.R first!

def plot_dimensions_required(
    ax, color, experiments, label, rsm=False, which_models=None
):
    xs = np.linspace(0.5, 1)
    dreq = dimensions_required(experiments, xs, rsm, which_models)
    ys = np.mean(dreq, axis=1)
    ystd = np.std(dreq, axis=1)
    ax.fill_between(xs, ys - ystd, ys + ystd, alpha=0.2, color=color, label=label)
    ax.plot(xs, ys, color=color)


with figure("p Value Comparison", figsize=[6.4, 6.4]) as f:
    A, B = f.subplots(2, 1)

    which_models = 10, 30
    for i, (group, exps) in enumerate(GROUP_EXPERIMENTS.items()):
        plot_dimensions_required(
            A, f"C{i}", exps, GROUP_NAME[group], which_models=which_models
        )
    plot_dimensions_required(
        A, "C5", ALL_EXPERIMENTS, "Shuffled", rsm=True, which_models=which_models
    )

    A.legend(loc="upper left")
    A.set_ylabel("Dimensions Required")
    A.set_ylim(1, 6)
    A.set_xlim(0.5, 1)
    A.set_xticks([])
    A.xaxis.set_major_formatter(PercentFormatter(1, 0))

    pval_df = pd.read_csv("pvalues.csv")
    sns.lineplot(data=pval_df, x="theta", y="p.value", hue="contrast", ax=B)
    B.set_yscale("log")
    B.axhline(0.05, c="k", ls=":", label="$p$ = 5\\%")
    B.set_xlim(0.5, 1)
    B.set_xlabel(r"Threshold $\theta$ (Percent Explained Variance)")
    B.set_ylabel("LMEM Significance $p$-value")
    B.legend(title="Comparison")

    f.align_ylabels([A, B])


# %%
# Possible new figure showing classification of backbone across models.

keep_vars = ["model", "sep_on_states", "sep_on_fr"]
melted = separability_df[keep_vars].melt(id_vars="model")

with figure("Backbone Classifiability Across Models") as f:
    ax = f.gca()
    sns.boxplot(
        data=melted,
        y="value",
        x="model",
        hue="variable",
        ax=ax,
    )
    ax.set_ylabel("Backbone Classification Accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0.5, 1)

# %%
# S21: showing that surrogate data has a linear manifold and real doesn't

with figure("Shuffled vs Real PCA", figsize=(7.5, 9)) as f:
    subfs = f.subfigures(4, 2)
    for i, (exp, subf) in enumerate(zip(GROUP_EXPERIMENTS["HO"], subfs.ravel())):
        subf.suptitle(f"Organoid {i+1}")
        axes = subf.subplots(
            1,
            2,
            sharex=True,
            sharey=True,
            gridspec_kw=dict(wspace=0, left=0.1, right=0.9, top=0.9, bottom=0.1),
        )
        for ax, rasters in zip(axes, [rasters_real, rasters_rsm]):
            raster = rasters[exp][0]
            model = rasters[exp][1][0]
            h = model.states(raster)
            points = model.pca.transform(raster._raster)[:, :2]
            ax.set_aspect("equal")
            ax.scatter(points[:, 1], points[:, 0], s=1, alpha=0.5, c=h, cmap="rainbow")


# %%
# S27: backbone units are less likely to be Poisson than non-rigid units

with figure("Backbone vs Non-Rigid Poisson Scores", figsize=(7, 2.5)) as f:
    ax = sns.boxplot(
        consistency_df,
        x="model",
        y="consistency",
        hue="backbone",
        width=0.5,
        ax=f.gca(),
        legend=False,
    )
    ax.set_xlabel("")
    labels = []
    for label in ax.get_xticklabels():
        model = label.get_text()
        subdf = consistency_df[consistency_df.model == model]
        labels.append(f"{model}\n({len(subdf)} Units)")
        print(label.get_text())
        p = stats.ks_2samp(
            subdf[subdf.backbone == "Backbone"].consistency,
            subdf[subdf.backbone == "Non-Rigid"].consistency,
            alternative="less",
        ).pvalue
        print(model, len(subdf), p)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of Significantly\nNon-Poisson States")
    ax.spines[['right', 'top']].set_visible(False)
