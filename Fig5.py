# Fig5.py
# Generate most of figure 5 of the final manuscript.
import pickle
import numpy as np
import matplotlib.pyplot as plt
import hmmsupport
from hmmsupport import get_raster, figure, load_metrics, Model, all_experiments
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

experiments = [
    ("organoid", exp) for exp in all_experiments("organoid") if exp.startswith("L")
] + [("eth", exp) for exp in all_experiments("eth")]

surr = "real"
hmm_library = "default"

figure_name = "Fig5"
if hmm_library != "default":
    figure_name += " " + hmm_library
if surr != "real":
    figure_name += " Surrogate"

plt.ion()
hmmsupport.figdir("paper")

bin_size_ms = 30
n_states = 10, 50
n_stateses = np.arange(n_states[0], n_states[-1] + 1)

print("Loading fitted HMMs and calculating entropy.")
srms = {}
with tqdm(total=len(experiments) * (1 + len(n_stateses))) as pbar:
    rasters = {}
    for source, exp in experiments:
        srms[exp] = load_metrics(exp)
        rasters[exp] = get_raster(source, exp, bin_size_ms, surr), []
        pbar.update()
        window = srms[exp]["burst_window"][0, :] / 1e3
        window[0] = min(window[0], -0.3)
        window[1] = max(window[1], 0.6)
        for n in n_stateses:
            rasters[exp][1].append(
                Model(
                    source,
                    exp,
                    bin_size_ms,
                    n,
                    surr,
                    library=hmm_library,
                    recompute_ok=False,
                )
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


try:
    with open(".cache/consistencies.pickle", "rb") as f:
        consistency_good, consistency_bad = pickle.load(f)
    print("Loaded consistency scores from file.")
except FileNotFoundError:
    print("Calculating consistency scores per neuron.")
    with tqdm(total=len(experiments) * len(n_stateses) * 2) as pbar:

        def consistency_scores(source, exp, n, surr):
            """
            Compute an n_states x n_units array indicating how likely a unit is
            to have nonzero firings in each time bin of a given state.
            """
            r = get_raster(source, exp, bin_size_ms, surr)
            m = Model(source, exp, bin_size_ms, n, surr)
            h = m.states(r)
            scores = np.array([(r._raster[h == i, :] > 0).mean(0) for i in range(n)])
            unit_order = srms[exp]["mean_rate_ordering"].flatten() - 1
            margins = rasters[exp][1][0].burst_margins
            state_order = r.state_order(h, margins, n_states=n)
            pbar.update()
            scores[np.isnan(scores)] = 0
            return scores[:, unit_order][state_order, :]

        consistency_good, consistency_bad = [
            {
                exp: [consistency_scores(source, exp, n, surr) for n in n_stateses]
                for (source, exp) in experiments
            }
            for surr in ["real", "rsm"]
        ]
    with open(".cache/consistencies.pickle", "wb") as f:
        pickle.dump((consistency_good, consistency_bad), f)


def separability(exp, X, pca=None, n_tries=100, validation=0.2):
    """
    Fit a linear classifier to the given features X and return its
    performance separating packet and non-packet units.
    """
    clf = SGDClassifier(n_jobs=12)
    if pca is not None and pca < X.shape[1]:
        clf = make_pipeline(PCA(n_components=pca), clf)
    y = ~(np.arange(X.shape[0]) < len(srms[exp]["non_scaf_units"]))
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


def separability_on_fr(r):
    """
    Check how well you can separate packet and non-packet units based on
    just their overall firing rates.
    """
    rates = r.rates("Hz").reshape((-1, 1))
    return separability(r.experiment, rates)


sep_on_states = {
    exp: [separability(exp, scores.T) for scores in scoreses]
    for exp, scoreses in consistency_good.items()
}

sep_on_fr = {exp: separability_on_fr(r) for exp, (r, _) in rasters.items()}


# %%

# The figure plots results for one example HMM first before comparing
# multiple, so pick a number of states and an experiment of interest.
source, exp = experiments[0]
n_states = 15
r, models = rasters[exp]
model = models[np.nonzero(n_stateses == n_states)[0][0]]

# Compute hidden states throughout the recording, and use them to identify
# which states happen at which peak-relative times.
h = model.states(r)
lmargin_h, rmargin_h = model.burst_margins
peaks = r.find_bursts(margins=model.burst_margins)
state_prob = r.observed_state_probs(h, burst_margins=model.burst_margins)
state_order = r.state_order(h, model.burst_margins, n_states=n_states)
poprate = r.coarse_rate()
unit_order = srms[exp]["mean_rate_ordering"].flatten() - 1
n_packet_units = len(srms[exp]["scaf_units"])

# inverse_unit_order[i] is the index of unit i in unit_order.
inverse_unit_order = np.zeros_like(unit_order)
inverse_unit_order[unit_order] = np.arange(len(unit_order))

# The figure compares three states of interest, which need to depend on the
# specific trained model we're looking at...
match source, hmm_library, surr, n_states:
    case "organoid", "default", "real", 15:
        interesting_states = [8, 9, 10]
    case "organoid", "default", "rsm", 15:
        interesting_states = [7, 8, 9]
    case "eth", "default", "rsm", 15:
        interesting_states = [4, 5, 6]
    case _:
        print("No interesting states chosen yet for these parameters.")
        interesting_states = state_prob[state_order, :].max(1).argsort()[-3:]

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
    for src, exp in experiments:
        for n in n_stateses:
            m = Model(src, exp, bin_size_ms, n, surr)
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


with figure(figure_name, figsize=(8.5, 11)) as f:
    # Subfigure A: example burst rasters.
    A = f.subplots(
        1,
        3,
        gridspec_kw=dict(wspace=0.1, top=0.995, bottom=0.85, left=0.04, right=0.95),
    )
    for ax, peak_float in zip(A, peaks):
        peak = int(round(peak_float))
        when = slice(peak + lmargin_h, peak + rmargin_h + 1)
        rsub = r._raster[when, :] / bin_size_ms
        hsub = np.array([np.nonzero(state_order == s)[0][0] for s in h[when]])
        t_sec = (np.ogrid[when] - peak) * bin_size_ms / 1000
        ax.imshow(
            hsub.reshape((1, -1)),
            cmap=alpha_rainbow,
            alpha=0.8,
            aspect="auto",
            vmin=0,
            vmax=n_states - 1,
            extent=[t_sec[0], t_sec[-1], 0.5, rsub.shape[1] + 0.5],
        )
        idces, times_ms = r.subtime(
            when.start * bin_size_ms, when.stop * bin_size_ms
        ).idces_times()
        times = (times_ms - (peak_float - when.start) * bin_size_ms) / 1000
        ax.plot(times, inverse_unit_order[idces] + 1, "ko", markersize=0.5)
        ax.set_ylim(0.5, rsub.shape[1] + 0.5)
        ax.set_xticks([0, 0.5])
        ax.set_xlim(*srms[exp]["burst_window"][0, :] / 1e3)
        ax.axhline(len(unit_order) - n_packet_units + 0.5, color="k", lw=0.5)
        ax.set_xlabel("Time from Peak (s)")
        ax.set_yticks([])
        ax2 = ax.twinx()
        when_ms = slice(when.start * bin_size_ms, when.stop * bin_size_ms)
        t_ms = np.ogrid[when_ms] - peak * bin_size_ms
        ax2.plot(t_ms / 1e3, poprate[when_ms], "r")
        ax2.set_ylim(0, 3)
        ax2.set_yticks([])

    ax2.set_ylabel("Population Rate (kHz)")
    ax2.set_yticks([0, 3])
    A[0].set_ylabel(
        r"Non-Packet \hspace{1.5em} Packet", y=1.03, horizontalalignment="right"
    )

    # Subfigure B: state heatmap.
    BCtop, BCbot = 0.78, 0.58
    ax = f.subplots(gridspec_kw=dict(top=BCtop, bottom=BCbot, left=0.06, right=0.3))
    im = ax.imshow(
        state_prob[state_order, :],
        vmin=0,
        vmax=1,
        extent=[t_sec[0], t_sec[-1], n_states + 0.5, 0.5],
        interpolation="none",
        aspect="auto",
    )
    ax.set_yticks([1, n_states])
    ax.set_xticks(0.3 * np.arange(-1, 3))
    ax.set_xlim(-0.3, 0.6)
    ax.set_xlabel("Time From Burst Peak (s)")
    ax.set_ylabel("Hidden State Number")
    plt.colorbar(
        im, ax=ax, label="Probability of Observing State", aspect=10, ticks=[0, 1]
    )

    # Subfigure C: state examples.
    Cleft, Cwidth = 0.35, 0.65
    (A, RA), (B, RB), (C, RC) = [
        f.subplots(
            1,
            2,
            gridspec_kw=dict(
                top=BCtop,
                bottom=BCbot,
                width_ratios=[3, 1],
                wspace=0,
                left=Cleft + Cwidth * l,
                right=Cleft + Cwidth * r,
            ),
        )
        for l, r in [(0.06, 0.26), (0.4, 0.61), (0.76, 0.96)]
    ]
    deltas = dBA, dCB = [
        f.subplots(
            gridspec_kw=dict(
                top=BCtop,
                bottom=BCbot,
                left=Cleft + Cwidth * l,
                right=Cleft + Cwidth * r,
            )
        )
        for l, r in [(0.305, 0.365), (0.655, 0.715)]
    ]

    examples = [A, B, C]
    rates = [RA, RB, RC]
    for ax in examples:
        ax.set_xticks([])
        ax.set_xlabel("Realizations", rotation=35)
    for ax in rates:
        ax.set_xlim([0, 5])
        ax.set_xticks([0, 5])
        ax.set_xlabel("FR (Hz)")
    for ax in deltas:
        ax.set_xticks([-3, 2])
        ax.set_xlim([-3, 2])
        ax.set_xlabel("$\\Delta$FR")
    for ax in examples + rates + deltas:
        ax.set_yticks([])
        ax.set_ylim(0.5, rsub.shape[1] + 0.5)
        ax.axhline(len(unit_order) - n_packet_units + 0.5, color="k", lw=0.5)
    A.set_ylabel(r"Non-Packet \hspace{3em} Packet", y=1.01, horizontalalignment="right")

    for axS, axH, s in zip(examples, rates, interesting_states):
        data = r._raster[h == state_order[s], :][:, unit_order]
        data_sub = data[np.random.choice(data.shape[0], 60), :]
        axS.set_title(f"State {s+1}")
        axS.imshow(
            data_sub.T,
            aspect="auto",
            interpolation="none",
            vmin=0,
            vmax=r._raster.max(),
            extent=[0, 1, r._raster.shape[1] + 0.5, 0.5],
        )

        axH.plot(
            data.mean(0),
            np.arange(r._raster.shape[1]) + 1,
            c=alpha_rainbow(s / (n_states - 1)),
        )

    for ax, s0, s1 in zip(deltas, interesting_states[:-1], interesting_states[1:]):
        mu0 = r._raster[h == state_order[s0], :].mean(0)
        mu1 = r._raster[h == state_order[s1], :].mean(0)
        delta = mu1 - mu0
        ax.plot(
            delta[unit_order], np.arange(r._raster.shape[1]) + 1, c="red", alpha=0.3
        )

    # Subfigure D: entropy specifically for Organoid 1.
    DEFtop, DEFbot = 0.51, 0.31
    en, pr = f.subplots(
        2,
        1,
        gridspec_kw=dict(
            height_ratios=[3, 2], top=DEFtop, bottom=DEFbot, left=0.06, right=0.31
        ),
    )

    lmargin, rmargin = model.burst_margins
    time_sec = np.arange(lmargin, rmargin + 1) * bin_size_ms / 1000
    ent = np.array([m.mean_entropy for m in rasters[exp][1]])
    meanent, stdent = ent.mean(0), ent.std(0)
    en.plot(time_sec, meanent - stdent, "C0", label=exp)
    en.plot(time_sec, meanent + stdent, "C0")
    en.plot(time_sec, meanent, "C0--")
    en.fill_between(time_sec, meanent - stdent, meanent + stdent, alpha=0.5, color="C0")

    for a in (en, pr):
        a.axvspan(*srms[exp]["scaf_window"][0, :] / 1e3, color="gray", alpha=0.3)
        a.set_xlim(srms[exp]["burst_window"][0, :] / 1e3)

    entropy_range = 3
    en.set_ylim(0, entropy_range)
    t_ms = np.arange(lmargin * bin_size_ms, (rmargin + 1) * bin_size_ms)
    for peak in peaks:
        peak_ms = int(round(peak * bin_size_ms))
        burst = poprate[
            peak_ms + lmargin * bin_size_ms : peak_ms + (rmargin + 1) * bin_size_ms
        ]
        pr.plot(t_ms / 1e3, burst, "C0", alpha=0.1)

    en.set_xticks([])
    pr.set_xticks([0, 0.5])
    en.set_ylabel("Entropy (bits)")
    en.set_yticks([0, entropy_range])
    pr.set_yticks([0, 2])
    pr.set_ylabel("Population Rate (Hz)")
    pr.set_xlabel("Time from Burst Peak (s)")
    f.align_ylabels((en, pr))

    # Subfigure E: PCA of real vs. surrogate data.
    E = f.subplots(
        1, 1, gridspec_kw=dict(top=DEFtop, bottom=DEFbot, left=0.35, right=0.6)
    )
    E.set_aspect("equal")
    data = pca_good.transform(r._raster)[:, 1::-1]
    E.scatter(data[:, 0], data[:, 1], s=2, c=model.states(r), cmap=alpha_rainbow)
    E.set_ylim([-3, 13])
    E.set_xlim([-3, 8])
    E.set_xlabel("PC2")
    E.set_xticks([0, 5])
    E.set_ylabel("PC1")
    E.set_yticks([0, 5, 10])

    # Subfigure F: explained variance ratio with inset.
    F = f.subplots(
        1,
        1,
        gridspec_kw=dict(top=DEFtop, bottom=DEFbot, wspace=0.4, left=0.64, right=0.98),
    )
    F.plot(np.arange(n_states) + 1, pca_good.explained_variance_ratio_, label="Real")
    F.plot(
        np.arange(n_states) + 1, pca_bad.explained_variance_ratio_, label="Randomized"
    )
    F.set_xticks([1, 15])
    F.set_xlabel("Principal Component")
    F.set_ylabel("Explained Variance Ratio")
    F.set_yticks([0, 1])
    F.legend(ncol=2, loc="upper right")
    bp = F.inset_axes([0.3, 0.2, 0.6, 0.6])
    for pve, x in zip([pve_good, pve_bad], [0.8, 1.2]):
        bp.violinplot(
            pve, showmedians=True, showextrema=False, positions=[x], widths=0.2
        )
    bp.set_ylim([0.38, 1.02])
    bp.set_xlim([0.6, 1.4])
    bp.set_xticks([])
    bp.set_yticks([])
    F.indicate_inset_zoom(bp, edgecolor="black")

    # Subfigure G: somehow show what's happening on the right.
    GHItop, GHIbot = 0.25, 0.04
    G = f.subplots(
        1, 1, gridspec_kw=dict(top=GHItop, bottom=GHIbot, left=0.03, right=0.3)
    )
    scores = consistency_good[exp][10]
    G.imshow(scores, aspect="auto", interpolation="none")
    G.set_xticks([])
    G.set_yticks([])
    G.set_xlabel(r"Non-Packet \hspace{2.5cm} Packet")
    G.set_ylabel("State")

    # Subfigure H: PCA of consistency scores for a single organoid, showing
    # that it's sufficient to separate packet/non-packet units.
    H = f.subplots(
        1, 1, gridspec_kw=dict(top=GHItop, bottom=GHIbot, left=0.35, right=0.6)
    )
    scores = consistency_good[exp][10]
    is_packet = ~(np.arange(scores.shape[1]) < len(srms[exp]["non_scaf_units"]))
    pca = PCA(n_components=2).fit_transform(scores.T)
    H.set_aspect("equal")
    H.scatter(pca[:, 1], pca[:, 0], c=is_packet)
    H.set_ylabel("PC1")
    H.set_xlabel("PC2")

    # Subfigure I: per-organoid separability metric.
    I = f.subplots(
        1, 1, gridspec_kw=dict(top=GHItop, bottom=GHIbot, left=0.67, right=0.98)
    )
    I.violinplot(
        sep_on_states.values(),
        positions=np.arange(len(experiments)),
        showextrema=False,
        showmeans=True,
    )
    I.plot([], [], "C0_", ms=10, label="By State Structure")
    I.plot(sep_on_fr.values(), "_", ms=10, label="By Firing Rate")
    I.set_xticks(
        range(len(experiments)), [f"Org.\\ {i+1}" for i in range(len(experiments))]
    )
    I.set_ylim([0.45, 1.05])
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    I.set_yticks(ticks, [f"{100*t:.0f}\\%" for t in ticks])
    I.set_ylabel("Packet / Non-Packet Separability")
    I.legend()
