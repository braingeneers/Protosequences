# PCAComparison
#
# Compare the variance explained by the first few principal components in
# each of the experiments under a given heading.
import itertools
import joblib
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

experiments = all_experiments(source)


def raster_valid(exp: str):
    try:
        return get_raster(source, exp, bin_size_ms).N > 15
    except Exception as e:
        print(exp, "failed to load:", e)
        return False


rasters: dict[str, hmmsupport.Raster] = {
    exp: get_raster(source, exp, bin_size_ms)
    for exp in tqdm(experiments, desc="Loading rasters")
    if raster_valid(exp)
}
experiments = list(rasters.keys())

print(len(experiments), "experiments have any data")

bad_rasters: dict[str, hmmsupport.Raster] = {
    exp: get_raster(source, exp, bin_size_ms, "rsm")
    for exp in tqdm(experiments, desc="Loading surrogate rasters")
}

models: dict[str, list[Model]] = {
    exp: joblib.Parallel(n_jobs=16)(
        joblib.delayed(Model)(source, exp, bin_size_ms, n, recompute_ok=False)
        for n in n_stateses
    )
    for exp in tqdm(experiments, desc="Loading models")
}

bad_models: dict[str, list[Model]] = {
    exp: joblib.Parallel(n_jobs=16)(
        joblib.delayed(Model)(source, exp, bin_size_ms, n, "rsm", recompute_ok=False)
        for n in n_stateses
    )
    for exp in tqdm(experiments, desc="Loading surrogate models")
}

are_ok = {
    k: np.logical_and(
        [m._hmm is not None for m in models[k]],
        [m._hmm is not None for m in bad_models[k]],
    )
    for k in experiments
}

models = {k: [m for m, ok in zip(models[k], are_ok[k]) if ok] for k in experiments}
bad_models = {
    k: [m for m, ok in zip(bad_models[k], are_ok[k]) if ok] for k in experiments
}

# %%

from sklearn.decomposition import PCA


def stateses(exp: str, bad: bool):
    ms = (bad_models if bad else models)[exp]
    return [np.exp(m._hmm.observations.log_lambdas) for m in ms]


def raster(exp: str, bad: bool = False):
    return (bad_rasters if bad else rasters)[exp]._raster


pcas = {exp: [PCA().fit(s) for s in stateses(exp, False)] for exp in experiments}
bad_pcas = {exp: [PCA().fit(s) for s in stateses(exp, True)] for exp in experiments}


def variance_by_axis(pca: PCA, exp: str, bad: bool = False):
    return pca.explained_variance_[:10]
    return np.var(pca.transform(raster(exp, bad)), axis=0)[:10]


tev = {
    exp: np.array([variance_by_axis(pca, exp) for pca in pcas[exp]])
    for exp in experiments
}
pev = {e: tev[e] / tev[e].sum(axis=1, keepdims=True) for e in experiments}

bad_tev = {
    exp: np.array([variance_by_axis(pca, exp, bad=True) for pca in bad_pcas[exp]])
    for exp in experiments
}
bad_pev = {e: bad_tev[e] / bad_tev[e].sum(axis=1, keepdims=True) for e in experiments}

if len(experiments) < 20:
    for exp in tqdm(experiments):
        with figure(exp, save_exts=[]) as f:
            ax = f.gca()
            component = 1 + np.arange(10)
            ax.plot(component, pev[exp].T, "C0")
            ax.plot(component, bad_pev[exp].T, "C1")
            ax.set_ylabel("Percent Explained Variance")
            ax.set_xlabel("Principal Component")


# %%
# Plot the number of components required to explain as much variance as the first
# dimension of the surrogate data.


def components_required(exp: str, thresh=None, bad=False):
    if thresh is None:
        thresh = bad_pev[exp][:, [0]]
    enough = np.cumsum((bad_pev if bad else pev)[exp], axis=1) > thresh
    return [
        np.argmax(enough[i, :]) + 1 if np.any(enough[i, :]) else enough.shape[1] + 1
        for i in range(enough.shape[0])
    ]


def plot_components_required(ax, experiments, get_label):
    ax.violinplot(
        [components_required(exp) for exp in experiments],
        showmeans=True,
        showextrema=False,
    )
    xs = np.arange(1, len(experiments) + 1)
    ax.set_xticks(xs, [get_label(i, e) for i, e in enumerate(experiments)])
    ax.set_ylabel("Components Required to Match Surrogate")


if source == "organoid":
    with figure("Components Required") as f:
        plot_components_required(f.gca(), experiments, lambda i, _: f"Organoid {i}")

elif source == "eth":
    with figure("Companents Required for ETH Organoids") as f:
        plot_components_required(f.gca(), experiments, lambda i, _: f"Well {i}")

elif source == "mouse":
    with figure("Mouse Dimensionality") as f:
        ax = f.gca()
        ax.hist([components_required(exp).mean() for exp in experiments])
        ax.set_xlabel("Mean Components Required Across Models")
        ax.set_ylabel("Number of Experiments")

elif source == "new_neuropixel":
    with figure("Components Required for Neuropixel") as f:
        plot_components_required(f.gca(), experiments, lambda _, e: e.split("_")[0])

elif source == "org_and_slice":
    for prefix in ["L", "M", "Pr"]:
        with figure(f"Components Required ({prefix}*)") as f:
            plot_components_required(
                f.gca(),
                [exp for exp in experiments if exp.startswith(prefix)],
                lambda _, e: e.split("_")[0],
            )


# %%
# Plot the dimensionality of each dataset as a function of the threshold for explained
# variance, comparing the surrogate data to the real data.


def pev_vs_thresholds(experiments, xs, bad=False):
    return np.array(
        [
            np.hstack(
                [components_required(exp, thresh=x, bad=bad) for exp in experiments]
            )
            for x in xs
        ]
    )


def plot_pev(ax, color, experiments, label, bad=False):
    xs = np.linspace(0.705, 1)
    pev = pev_vs_thresholds(experiments, xs, bad=bad)
    ys = pev.mean(axis=1)
    ystd = pev.std(axis=1)
    ax.fill_between(xs, ys - ystd, ys + ystd, alpha=0.2, color=color, label=label)
    ax.plot(xs, ys, color=color)


def ks_compare_pev(As, Bs):
    xs = np.linspace(0.7, 1)
    Apev = pev_vs_thresholds(As, xs)
    Bpev = pev_vs_thresholds(Bs, xs)
    kses = [stats.ks_2samp(Apev[i, :], Bpev[i, :]) for i in range(len(xs))]
    statistic = np.mean([ks.statistic for ks in kses])
    pvalue = stats.gmean([ks.pvalue for ks in kses])
    return statistic, pvalue


if source == "org_and_slice":
    group_name = dict(L="Organoid", M="Mouse", Pr="Primary")
    with figure("Dimensionality vs Threshold") as f:
        ax = f.gca()
        for i, prefix in enumerate(["L", "M", "Pr"]):
            expsub = [exp for exp in experiments if exp.startswith(prefix)]
            plot_pev(ax, f"C{i}", expsub, group_name[prefix])
        plot_pev(ax, "red", experiments, "Surrogate", True)
        ax.legend(loc="upper left")
        ax.set_xlabel("Explained Variance Threshold")
        ax.set_ylabel("Dimensions Required")
        ax.set_ylim(1, 6)
        ax.set_xlim(0.7, 1)

    for a, b in itertools.combinations(group_name, 2):
        effect, pvalue = ks_compare_pev(
            [exp for exp in experiments if exp.startswith(a)],
            [exp for exp in experiments if exp.startswith(b)],
        )
        print(
            f"{group_name[a]} vs {group_name[b]}: {effect:.2f}, p = {pvalue*100:.2e}%"
        )

# %%


def transformed_data(exp: str, bad: bool):
    pca = (bad_pcas if bad else pcas)[exp][state_idx]
    return pca.transform(raster(exp, bad))[:, :3].T


def transformed_states(exp: str, bad: bool):
    pca = (bad_pcas if bad else pcas)[exp][state_idx]
    return pca.transform(stateses(exp, bad)[state_idx])[:, :3].T


state_idx = 0
if source == "mouse":
    # These are the key experiments Mattia pointed out as having good
    # temporal structure.
    key_exps = ["1009-3", "1005-1", "366-2"]
    with figure("Key Mouse Surrogate Comparison", figsize=(4 * len(key_exps), 6)) as f:
        axes = f.subplots(
            2, len(key_exps), squeeze=False, subplot_kw=dict(projection="3d")
        )
        for j, exp in enumerate(key_exps):
            axes[0, j].set_title(f"{exp}: {rasters[exp].N} units")
            for i, bad in enumerate([False, True]):
                axes[i, j].plot(*transformed_data(exp, bad), color="grey", lw=0.1)
                axes[i, j].plot(*transformed_states(exp, bad), "o")

elif source == "new_neuropixel":
    key_exps = ["rec0_curated", "rec2_curated", "rec6_curated"]
    with figure("Neuropixel Surrogate Comparison", figsize=(4 * len(key_exps), 6)) as f:
        axes = f.subplots(
            2, len(key_exps), squeeze=False, subplot_kw=dict(projection="3d")
        )
        for j, exp in enumerate(key_exps):
            axes[0, j].set_title(f"{exp}: {rasters[exp].N} units")
            for i, bad in enumerate([False, True]):
                axes[i, j].plot(*transformed_data(exp, bad), color="grey", lw=0.1)
                axes[i, j].plot(*transformed_states(exp, bad), "o")

elif source == "org_and_slice":
    key_exps = ["L2", "M2S2", "Pr2"]
    with figure("Surrogate Comparison", figsize=(4 * len(key_exps), 6)) as f:
        axes = f.subplots(
            2, len(key_exps), squeeze=False, subplot_kw=dict(projection="3d")
        )
        for j, expname in enumerate(key_exps):
            exp = expname + "_t_spk_mat_sorted"
            axes[0, j].set_title(f"{expname}: {rasters[exp].N} units")
            for i, bad in enumerate([False, True]):
                axes[i, j].plot(*transformed_data(exp, bad), color="grey", lw=0.1)
                axes[i, j].plot(*transformed_states(exp, bad), "o")
