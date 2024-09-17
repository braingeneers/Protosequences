# Fig7.py
# Generate my two rows of figure 7 of the final manuscript.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from hmmsupport import Model, figure, get_raster, load_metrics

source = "org_and_slice"
group_name = {"HO": "Organoid", "MO": "Mouse Organoid", "MS": "Slice", "Pr": "Primary"}
name_group = {v: k for k, v in group_name.items()}
groups = {
    "HO": [
        "L1_t_spk_mat_sorted",
        "L2_7M_t_spk_mat_sorted",
        "L3_7M_t_spk_mat_sorted",
        "L5_t_spk_mat_sorted",
        "well1_t_spk_mat_sorted",
        "well4_t_spk_mat_sorted",
        "well5_t_spk_mat_sorted",
        "well6_t_spk_mat_sorted",
    ],
    "MO": [
        "MO1_t_spk_mat_sorted",
        "MO2_t_spk_mat_sorted",
        "MO3_t_spk_mat_sorted",
        "MO4_t_spk_mat_sorted",
        "MO5_t_spk_mat_sorted",
        "MO6_t_spk_mat_sorted",
        "MO7_t_spk_mat_sorted",
        "MO8_t_spk_mat_sorted",
        "MO9_t_spk_mat_sorted",
    ],
    "MS": [
        "M1S1_t_spk_mat_sorted",
        "M1S2_t_spk_mat_sorted",
        "M2S1_t_spk_mat_sorted",
        "M2S2_t_spk_mat_sorted",
        "M3S1_t_spk_mat_sorted",
        "M3S2_t_spk_mat_sorted",
    ],
    "Pr": [
        "Pr1_t_spk_mat_sorted",
        "Pr2_t_spk_mat_sorted",
        "Pr3_t_spk_mat_sorted",
        "Pr4_t_spk_mat_sorted",
        "Pr5_t_spk_mat_sorted",
        "Pr6_t_spk_mat_sorted",
        "Pr7_t_spk_mat_sorted",
        "Pr8_t_spk_mat_sorted",
    ],
}
exp_rms = (
    {exp: 5.0 for exp in groups["HO"]}
    | {exp: 3.0 for exp in groups["MO"]}
    | {exp: 6.0 for exp in groups["MS"]}
    | {exp: 3.0 for exp in groups["Pr"]}
    | {"MO1_t_spk_mat_sorted": 2.5, "MO2_t_spk_mat_sorted": 2.0}
)
experiments = [exp for exps in groups.values() for exp in exps]
short_name = {
    exp: group + str(i+1)
    for group, exps in groups.items()
    for i, exp in enumerate(exps)
}
exp_to_group = {
    exp: group_name[group] for group, exps in groups.items() for exp in exps
}

plt.ion()

bin_size_ms = 30
n_stateses = np.arange(10, 51)


backbone, nonrigid, unit_orders = {}, {}, {}
which_metrics = ["scaf_units", "non_scaf_units", "mean_rate_ordering"]
print("Loading metrics files.")
for exp in tqdm(experiments):
    metrics = load_metrics(exp, only_include=which_metrics)
    backbone[exp] = np.int32(metrics["scaf_units"].ravel()) - 1
    nonrigid[exp] = np.int32(metrics["non_scaf_units"].ravel()) - 1
    unit_orders[exp] = np.int32(metrics["mean_rate_ordering"].flatten()) - 1


print("Loading real and surrogate rasters and doing PCA on HMMs.")
with tqdm(total=2 * len(experiments) * (1 + len(n_stateses))) as pbar:
    rasters_real, rasters_rsm = {}, {}
    _rs = dict(real=rasters_real, rsm=rasters_rsm)
    for exp in experiments:
        for surr in ["real", "rsm"]:
            r, ms = _rs[surr][exp] = get_raster(source, exp, bin_size_ms, surr), []
            r.burst_rms = exp_rms[exp]
            r.unit_order = unit_orders[exp]
            pbar.update()
            for n in n_stateses:
                m = Model(source, exp, bin_size_ms, n, surr)
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


def separability(exp, X, pca=None, n_tries=100, validation=0.2):
    """
    Fit a linear classifier to the given features X and return its
    performance separating packet and non-packet units.
    """
    clf = SGDClassifier(n_jobs=12)
    if pca is not None and pca < X.shape[1]:
        clf = make_pipeline(PCA(n_components=pca), clf)
    y = ~(np.arange(X.shape[0]) < len(nonrigid[exp]))
    if validation is None:
        Xt = Xv = X
        yt = yv = y
    best, res = 0, 0
    for _ in range(n_tries):
        if validation is not None:
            Xt, Xv, yt, yv = train_test_split(X, y, stratify=y, test_size=validation)
        score = clf.fit(Xt, yt).score(Xv, yv)
        if score > best:
            best, res = score, clf.score(X, y)
    return res


consistencies = {
    e: mean_consistency(all_the_scores(e, True, rsm=False)) for e in tqdm(experiments)
}

consistency_df = pd.DataFrame(
    [
        dict(
            experiment=exp,
            model=exp_to_group[exp],
            consistency=c,
            label=label,
            backbone="Backbone" if label else "Non-Rigid",
        )
        for exp in experiments
        for label, unitgroup in enumerate([nonrigid, backbone])
        for c in consistencies[exp][unitgroup[exp]]
    ]
)


sep_on_states = {
    exp: [separability(exp, m.consistency.T) for m in ms]
    for exp, (_, ms) in rasters_real.items()
}

sep_on_fr = {
    exp: separability(exp, r.rates("Hz").reshape((-1, 1)))
    for exp, (r, _) in rasters_real.items()
}

separability_df = pd.DataFrame(
    [
        dict(group=exp_to_group[exp], value=value, by="State Structure")
        for exp, values in sep_on_states.items()
        for value in values
    ]
    + [
        dict(group=exp_to_group[exp], value=value, by="Firing Rate")
        for exp, value in sep_on_fr.items()
    ]
)

pd.DataFrame(
    [
        dict(
            sample_type=name_group[exp_to_group[exp]],
            sample_id=short_name[exp],
            K=n_stateses[i],
            value=(value - sep_on_fr[exp]) / (1 - sep_on_fr[exp]),
        )
        for exp, values in sep_on_states.items()
        for i, value in enumerate(values)
    ]
).to_csv("separability.csv", index=False)

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


with figure("Fig7", figsize=(8.5, 3.0), save_exts=["png", "svg"]) as f:
    G, H = f.subplots(1, 2, gridspec_kw=dict(width_ratios=[2, 3]))

    # Subfigure G: dimensionality as a function of PC inclusion threshold.
    which_models = 10, 30
    for i, (group, exps) in enumerate(groups.items()):
        plot_dimensions_required(
            G, f"C{i}", exps, group_name[group], which_models=which_models
        )
    plot_dimensions_required(
        G, "C5", experiments, "Shuffled", rsm=True, which_models=which_models
    )
    G.legend(loc="upper left")
    G.set_xlabel(r"Threshold $\theta$ (Percent Explained Variance)")
    G.set_ylabel("Dimensions Required")
    G.set_ylim(1, 6)
    G.set_xlim(0.7, 1)
    G.xaxis.set_major_formatter(PercentFormatter(1, 0))

    # Subfigure H:
    xs = np.linspace(0.5, 1, 100)
    for model in group_name.values():
        y_bb = fraction_above_xs(xs, consistency_df, True, model)
        y_nr = fraction_above_xs(xs, consistency_df, False, model)
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
    H.set_ylabel("Fraction of Non-Poisson States by Unit")


# %%
# Possible new figure showing classification of backbone across models.

with figure("Backbone Classifiability Across Models") as f:
    ax = f.gca()
    sns.boxplot(
        data=separability_df,
        y="value",
        x="group",
        hue="by",
        ax=ax,
    )
    ax.set_ylabel("Backbone Classification Accuracy")
    ax.set_xlabel("")
    ax.legend()

# %%
# S21: showing that surrogate data has a linear manifold and real doesn't

with figure("Shuffled vs Real PCA", figsize=(7.5, 9)) as f:
    subfs = f.subfigures(4, 2)
    for i, (exp, subf) in enumerate(zip(groups["HO"], subfs.ravel())):
        subf.suptitle(f"Organoid {i}")
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
# S25: dimensionality as a function of PC inclusion threshold

which_models = 10, 50
dimensions = {}
xs = np.linspace(0, 1, num=100)[1:]
for i, (group, exps) in enumerate(groups.items()):
    dimensions[group] = dimensions_required(exps, xs, which_models=which_models)
dimensions["*"] = dimensions_required(
    experiments, xs, which_models=which_models, rsm=True
)

for a, b in [("HO", "MS"), ("MS", "Pr"), ("HO", "Pr")]:
    scores = stats.mannwhitneyu(dimensions[a], dimensions[b], axis=1).pvalue
    print(a, b, stats.gmean(scores[~np.isnan(scores)]))


with figure("Fig 7G Expanded") as f:
    ax = f.gca()
    which_models = 10, 50
    for i, prefix in enumerate(group_name):
        expsub = [e for e in experiments if e.startswith(prefix)]
        plot_dimensions_required(
            ax, f"C{i}", expsub, group_name[prefix], which_models=which_models
        )
    plot_dimensions_required(
        ax, "red", experiments, "Shuffled", rsm=True, which_models=which_models
    )
    ax.legend(loc="upper left")
    ax.set_xlabel("Explained Variance Threshold")
    ax.set_ylabel("Dimensions Required")
    ax.set_ylim(1, 6)
    ax.set_xlim(0.7, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1, 0))


# %%
# S27: backbone units are less likely to be Poisson than non-rigid units

with figure("Backbone vs Non-Rigid Poisson Scores") as f:
    ax = sns.boxplot(
        consistency_df,
        x="model",
        y="consistency",
        hue="backbone",
        width=0.5,
        ax=f.gca(),
    )
    ax.set_xlabel("")
    labels = []
    for label in ax.get_xticklabels():
        model = label.get_text()
        subdf = consistency_df[consistency_df.model == model]
        labels.append(f"{model} ($n = {len(subdf)}$)")
        print(label.get_text())
        p = stats.ks_2samp(
            subdf[subdf.backbone == "Backbone"].consistency,
            subdf[subdf.backbone == "Non-Rigid"].consistency,
            alternative="less",
        ).pvalue
        print(model, len(subdf), p)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of Non-Poisson States by Unit")
    ax.legend()
