# Fig7.py
# Generate my two rows of figure 7 of the final manuscript.
import numpy as np
import matplotlib.pyplot as plt
import hmmsupport
from hmmsupport import get_raster, figure, load_metrics, Model, all_experiments
from sklearn.decomposition import PCA
from tqdm import tqdm

source = "org_and_slice"
experiments = [exp for exp in all_experiments(source) if exp.startswith("L")]

plt.ion()
hmmsupport.figdir("paper")

bin_size_ms = 30
n_states = 10, 20
n_stateses = np.arange(n_states[0], n_states[-1] + 1)

print("Loading fitted HMMs and calculating entropy.")
srms = {}
with tqdm(total=len(experiments) * (1 + len(n_stateses))) as pbar:
    rasters = {}
    for exp in experiments:
        srms[exp] = load_metrics(exp)
        rasters[exp] = get_raster(source, exp, bin_size_ms), []
        pbar.update()
        window = srms[exp]["burst_window"].ravel() / 1e3
        window[0] = min(window[0], -0.3)
        window[1] = max(window[1], 0.6)
        for n in n_stateses:
            rasters[exp][1].append(
                Model(source, exp, bin_size_ms, n, recompute_ok=False)
            )
            rasters[exp][1][-1].compute_entropy(rasters[exp][0], *window)
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
exp = experiments[0]
n_states = 15
r, models = rasters[exp]
model = rasters[exp][1][np.nonzero(n_stateses == n_states)[0][0]]

# Fit PCA to the states of the model up there as well as the same-parameter
# model trained on the surrogate data.
r_bad = get_raster(source, exp, bin_size_ms, "rsm")
model_bad = Model(source, exp, bin_size_ms, n_states, "rsm")
pca_good, pca_bad = [
    PCA().fit(np.exp(m._hmm.observations.log_lambdas)) for m in [model, model_bad]
]


# Do a PCA like that for every trained model across all n_stateses and all
# organoids, and store the fraction of variance explained by PC1 in two
# arrays, one for the real and one for the surrogate data.
def pve_score(surr):
    scores = []
    for exp in experiments:
        for n in n_stateses:
            m = Model(source, exp, bin_size_ms, n, surr)
            pca = PCA().fit(np.exp(m._hmm.observations.log_lambdas))
            scores.append(pca.explained_variance_ratio_[0])
    return scores


pve_good, pve_bad = [np.array(pve_score(surr)) for surr in ["real", "rsm"]]

# Create a color map which is identical to gist_rainbow but with all the
# colors rescaled by 0.3 to make them match the ones above that are
# affected by alpha.
alpha_rainbow = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "alpha_rainbow", 0.5 + 0.5 * plt.get_cmap("gist_rainbow")(np.linspace(0, 1, 256))
)


with figure("Fig7", figsize=(8.5, 5.5)) as f:
    ABCtop, ABCbot = 0.95, 0.5
    # Subfigure A: PCA of real vs. surrogate data.
    A = f.subplots(
        1, 1, gridspec_kw=dict(top=ABCtop, bottom=ABCbot, left=0.35, right=0.6)
    )
    A.set_aspect("equal")
    data = pca_good.transform(r._raster)[:, 1::-1]
    A.scatter(data[:, 0], data[:, 1], s=2, c=model.states(r), cmap=alpha_rainbow)
    A.set_ylim([-3, 13])
    A.set_xlim([-3, 8])
    A.set_xlabel("PC2")
    A.set_xticks([0, 5])
    A.set_ylabel("PC1")
    A.set_yticks([0, 5, 10])

    # Subfigure F: explained variance ratio with inset.
    B = f.subplots(
        1,
        1,
        gridspec_kw=dict(top=ABCtop, bottom=ABCbot, wspace=0.4, left=0.64, right=0.98),
    )
    B.plot(np.arange(n_states) + 1, pca_good.explained_variance_ratio_, label="Real")
    B.plot(
        np.arange(n_states) + 1, pca_bad.explained_variance_ratio_, label="Randomized"
    )
    B.set_xticks([1, 15])
    B.set_xlabel("Principal Component")
    B.set_ylabel("Explained Variance Ratio")
    B.set_yticks([0, 1])
    B.legend(ncol=2, loc="upper right")
    bp = B.inset_axes([0.3, 0.2, 0.6, 0.6])
    for pve, x in zip([pve_good, pve_bad], [0.8, 1.2]):
        bp.violinplot(
            pve, showmedians=True, showextrema=False, positions=[x], widths=0.2
        )
    bp.set_ylim([0.38, 1.02])
    bp.set_xlim([0.6, 1.4])
    bp.set_xticks([])
    bp.set_yticks([])
    B.indicate_inset_zoom(bp, edgecolor="black")
