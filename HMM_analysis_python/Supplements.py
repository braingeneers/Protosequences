# Supplements.py
# Generate supplementary figures 22, 23, 24, 26, 28, 32, 33D, and 34, as well as some
# data files used for statistical analyses (and figure 7F).
import numpy as np
import pandas as pd
import seaborn as sns
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
    cv_binsize_df,
    cv_plateau_df,
    figure,
    get_raster,
    load_metrics,
    state_traversal_df,
)

EXP_RMS = (
    {exp: 5.0 for exp in GROUP_EXPERIMENTS["HO"]}
    | {exp: 3.0 for exp in GROUP_EXPERIMENTS["MO"]}
    | {exp: 6.0 for exp in GROUP_EXPERIMENTS["MS"]}
    | {exp: 3.0 for exp in GROUP_EXPERIMENTS["Pr"]}
    | {"MO1_t_spk_mat_sorted": 2.5, "MO2_t_spk_mat_sorted": 2.0}
)

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


# Quick data loading sanity check.
for exp, (r_real, _) in rasters_real.items():
    nunits = r_real._raster.shape[1]
    meanfr = r_real._raster.mean() / r_real.bin_size_ms * 1000
    nbursts = len(r_real.find_bursts())
    print(
        f"{exp} has {nunits} units firing at {meanfr:0.2f} " f"Hz with {nbursts} bursts"
    )


# Vectors of the first ten explained variance ratios for all of the experiments.
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


def unit_consistency(model, raster):
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

    peaks, bounds = raster.find_burst_edges()
    edges = np.int64(np.round((peaks[:, None] + bounds) / raster.bin_size_ms))
    ok = np.zeros_like(h, dtype=bool)
    for start, end in edges:
        ok[start:end] = True
    h[~ok] = -1

    rsubs = [raster._raster[h == state, :] for state in range(K)]
    ret = [poisson_test(rsub) for rsub in rsubs]
    return np.array(ret), np.array([rsub.shape[0] for rsub in rsubs])


def mean_consistency(exp):
    """
    Returns a vector of shape (n_units,) giving the fraction of all states across all
    models where the unit is too consistent to be Poisson, weighted by how frequently
    that state was observed. Units which never fire in a given state are considered
    trivially Poisson.
    """
    # Gather consistency scores and observation counts across all states of all models.
    r, models = rasters_real[exp]
    scores_nobs = [unit_consistency(model, r) for model in models]
    scores = np.vstack([s for s, _ in scores_nobs])
    nobs = np.hstack([n for _, n in scores_nobs])
    # Compare in the correct sense so that NaNs are treated as "not
    # consistent", i.e. potentially Poisson, then invert.
    return np.ma.average(scores < 0.01, axis=0, weights=nobs)


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
        for c in mean_consistency(exp)[unitgroup[exp]]
    ]
)

# This is used by lmem.R to generate the p-values for S32.
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

# This subset is used to generate 7F.
dreq_df[dreq_df.theta == 0.75].to_csv("dimensions_0.75.csv", index=False)


# %%
# S22: Cross-validation by bin size.

cv_binsize = cv_binsize_df()

with figure("Cross-Validation by Bin Size") as f:
    ax = f.gca()
    sns.boxplot(
        data=cv_binsize,
        x="bin_size",
        y="total_delta_ll",
        ax=ax,
    )
    ax.set_ylabel("Total $\\Delta$ Log Likelihood Real vs. Shuffled")
    ax.set_xlabel("Bin Size (ms)")
    ax.set_yscale("log")


# %%
# S23: The plateau that occurs above 10 states.

cv_plateau = cv_plateau_df()
cv_plateau["short_label"] = cv_plateau.experiment.map(SHORT_NAME.get)


with figure("Cross-Validation Plateau") as f:
    ax = sns.lineplot(
        data=cv_plateau,
        x="states",
        y="total_delta_ll",
        hue="short_label",
        errorbar="sd",
    )
    ax.set_ylabel("Total $\\Delta$ Log Likelihood Real vs. Shuffled")
    ax.set_xlabel("Number of Hidden States in Model")
    ax.legend(title=None, ncol=2)
    ax.set_yscale("log")


# S24: Cross-validation proving that the model performance is better for
# the real data than for the shuffled data.

subdata = cv_plateau[(cv_plateau.states >= 10) & (cv_plateau.states <= 30)]
melted = pd.melt(
    subdata, id_vars=["short_label"], value_vars=["total_ll", "total_surr_ll"]
)
melted.variable = melted.variable.map(
    dict(total_ll="Real", total_surr_ll="Shuffled").get
)

with figure("Overall Model Validation", figsize=(6.4, 6.4)) as f:
    raw, delta = f.subplots(2, 1)
    sns.boxplot(
        data=melted,
        x="short_label",
        hue="variable",
        y="value",
        width=1.0,
        ax=raw,
    )
    raw.set_ylabel("Total Log Likelihood")
    raw.legend()
    raw.set_xlabel("")
    sns.boxplot(data=subdata, x="short_label", y="total_delta_ll", ax=delta)
    delta.set_ylabel("Total $\\Delta$ Log Likelihood Real vs. Shuffled")
    delta.set_xlabel("")
    delta.set_yscale("log")
    f.align_ylabels()


# %%
# S28: surrogate data has a linear manifold, whereas real data is more complicated.

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
# S32: supplemement to figure 7F showing the dimensions required and the significance as
# a function of threshold θ.
#
# Make sure to generate dimensions.csv and run lmem.R first!


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
# S33D: backbone units are less likely to be Poisson than non-rigid units

with figure("Poisson Test", figsize=(7, 2.5), save_exts=["png", "svg"]) as f:
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
    ax.spines[["right", "top"]].set_visible(False)


# %%
# S26: State traversal by number of states.

traversal = state_traversal_df()
traversal.to_csv("traversal.csv")

with figure("States Traversed by K") as f:
    ax = sns.lineplot(
        traversal,
        x="K",
        y="rate",
        ax=f.gca(),
        hue="sample_type",
        errorbar="sd",
    )
    ax.set_ylabel("Average States Traversed in Per Second in Backbone Window")
    ax.set_xlabel("Number of Hidden States")
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(0, 30)
    ax.set_xticks([10, 15, 20, 25, 30])
    reg = stats.linregress(traversal.K, traversal.rate)
    x = np.array([9.5, 30.5])
    ax.plot(
        x,
        reg.intercept + reg.slope * x,
        color="k",
        linestyle="--",
        label=f"Trendline ($r^2 = {reg.rvalue**2:.2}$)",
    )
    ax.legend(loc="lower right")

# S34: State traversal by model.

with figure("States Traversed by Model") as f:
    bins = np.arange(1, 38, 3)
    axes = f.subplots(4, 1)
    for (i, ax), (model, dfsub) in zip(
        enumerate(axes), traversal.groupby("sample_type")
    ):
        sns.histplot(
            dfsub,
            x="rate",
            color=f"C{i}",
            kde=True,
            linewidth=0,
            kde_kws=dict(cut=0),
            ax=ax,
            bins=bins,
            label=model,
        )
        ax.set_xlim(0, 40)
        if i == len(axes) - 1:
            ax.set_xlabel("Average States Traversed in Per Second in Backbone Window")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
    f.legend(loc=(0.775, 0.5))
