# Fig7.py
# Generate my two rows of figure 7 of the final manuscript.
import numpy as np
import matplotlib.pyplot as plt
import hmmsupport
from hmmsupport import get_raster, figure, load_metrics, Model, all_experiments
from sklearn.decomposition import PCA
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter

source = "org_and_slice"
experiments = all_experiments(source)

plt.ion()
hmmsupport.figdir("paper")

bin_size_ms = 30
n_stateses = np.arange(10, 21)

print("Loading real and surrogate rasters and doing PCA on HMMs.")
srms = {}
with tqdm(total=2 * len(experiments) * (1 + len(n_stateses))) as pbar:
    rasters = {}
    rasters_bad = {}
    _rs = dict(real=rasters, rsm=rasters_bad)
    for exp in experiments:
        srms[exp] = load_metrics(exp)
        window = srms[exp]["burst_window"].ravel() / 1e3
        window[0] = min(window[0], -0.3)
        window[1] = max(window[1], 0.6)
        for surr in ["real", "rsm"]:
            _rs[surr][exp] = get_raster(source, exp, bin_size_ms, surr), []
            pbar.update()
            for n in n_stateses:
                m = Model(source, exp, bin_size_ms, n, surr, recompute_ok=False)
                m.pca = PCA().fit(np.exp(m._hmm.observations.log_lambdas))
                _rs[surr][exp][1].append(m)
                pbar.update()

for k, (r, _) in rasters.items():
    nunits = r._raster.shape[1]
    meanfr = r._raster.mean() / r.bin_size_ms * 1000
    nbursts = len(r.find_bursts())
    print(
        f"{k} has {nunits} units firing at {meanfr:0.2f} " f"Hz with {nbursts} bursts"
    )


# %%

# The figure plots results for one example HMM first before comparing
# multiple, so pick a number of states and an experiment of interest.
exp = "L1_t_spk_mat_sorted"
n_states = 15
n_states_index = np.nonzero(n_stateses == n_states)[0][0]
r, models = rasters[exp]
model = rasters[exp][1][n_states_index]
r_bad, models_bad = rasters_bad[exp]
model_bad = rasters_bad[exp][1][n_states_index]


# Vectors of the first ten explained variance ratios for all of the
# experiments.
pve, pve_bad = [
    {
        e: np.array([m.pca.explained_variance_ratio_[:10] for m in ms])
        for e, (_, ms) in rs.items()
    }
    for rs in [rasters, rasters_bad]
]


def components_required(exp, thresh, bad=False):
    enough = np.cumsum((pve_bad if bad else pve)[exp], axis=1) > thresh
    return [
        np.argmax(enough[i, :]) + 1 if np.any(enough[i, :]) else enough.shape[1] + 1
        for i in range(enough.shape[0])
    ]


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


# Create a color map which is identical to gist_rainbow but with all the
# colors rescaled by 0.3 to make them match the ones above that are
# affected by alpha.
alpha_rainbow = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "alpha_rainbow", 0.5 + 0.5 * plt.get_cmap("gist_rainbow")(np.linspace(0, 1, 256))
)


with figure("Fig7", figsize=(8.5, 5.5)) as f:
    A, B, C = f.subplots(
        1,
        3,
        gridspec_kw=dict(
            top=0.95, bottom=0.55, left=0.06, right=0.97, width_ratios=[3, 5, 5]
        ),
    )
    # Subfigure A: PCA of real vs. surrogate data.
    A.set_aspect("equal")
    data = model.pca.transform(r._raster)[:, 1::-1]
    A.scatter(data[:, 0], data[:, 1], s=2, c=model.states(r), cmap=alpha_rainbow)
    A.set_ylim([-3, 13])
    A.set_xlim([-3, 8])
    A.set_xlabel("PC2")
    A.set_xticks([0, 5])
    A.set_ylabel("PC1")
    A.set_yticks([0, 5, 10])

    # Subfigure B: explained variance ratio with inset.
    states_upto = 10
    B.plot(
        np.arange(states_upto) + 1,
        model.pca.explained_variance_ratio_[:states_upto],
        label="Real",
    )
    B.plot(
        np.arange(states_upto) + 1,
        model_bad.pca.explained_variance_ratio_[:states_upto],
        label="Random",
    )
    B.set_xticks([1, states_upto])
    B.set_xlabel("Principal Component")
    B.set_ylabel("Explained Variance Ratio")
    B.set_yticks([0, 1])
    B.legend(ncol=2, loc="upper right")
    bp = B.inset_axes([0.3, 0.2, 0.6, 0.6])
    for ps, x in zip([pve[exp][:, 0], pve_bad[exp][:, 0]], [0.8, 1.2]):
        bp.violinplot(
            ps, showmedians=True, showextrema=False, positions=[x], widths=0.2
        )
    bp.set_ylim([0.38, 1.02])
    bp.set_xlim([0.6, 1.4])
    bp.set_xticks([])
    bp.set_yticks([])
    B.indicate_inset_zoom(bp, edgecolor="black")

    # Subfigure C: dimensionality as a function of PC inclusion threshold.
    groups = dict(L="Organoid", M="Mouse", Pr="Primary")
    for i, prefix in enumerate(groups):
        expsub = [e for e in experiments if e.startswith(prefix)]
        plot_pev(C, f"C{i}", expsub, groups[prefix])
    plot_pev(C, "red", experiments, "Surrogate", bad=True)
    C.legend(loc="upper left")
    C.set_xlabel("Explained Variance Threshold")
    C.set_ylabel("Dimensions Required")
    C.set_ylim(1, 6)
    C.set_xlim(0.7, 1)
    C.xaxis.set_major_formatter(PercentFormatter(1, 0))
