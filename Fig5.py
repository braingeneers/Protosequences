# Fig5.py
# Generate figure 5 of the final manuscript.
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

source = "org_and_slice"
experiments = [exp for exp in all_experiments(source) if exp.startswith("L")]

plt.ion()
hmmsupport.figdir("paper")

bin_size_ms = 30
n_stateses = np.arange(10, 51)

print("Loading fitted HMMs and calculating consistency.")
metricses = {}
with tqdm(total=2 * len(experiments) * (1 + len(n_stateses))) as pbar:
    rasters_real, rasters_rsm = {}, {}
    rasterses = dict(real=rasters_real, rsm=rasters_rsm)
    for exp in experiments:
        metricses[exp] = load_metrics(exp)
        window = metricses[exp]["burst_window"].ravel() / 1e3
        window[0] = min(window[0], -0.3)
        window[1] = max(window[1], 0.6)
        for surr, rs in rasterses.items():
            rs[exp] = get_raster(source, exp, bin_size_ms), []
            pbar.update()
            for n in n_stateses:
                m = Model(source, exp, bin_size_ms, n, recompute_ok=False)
                m.compute_consistency(rs[exp][0], metricses[exp])
                rs[exp][1].append(m)
                pbar.update()

for k, (r, _) in rasters_real.items():
    nunits = r._raster.shape[1]
    meanfr = r._raster.mean() / r.bin_size_ms * 1000
    nbursts = len(r.find_bursts())
    print(
        f"{k} has {nunits} units firing at {meanfr:0.2f} " f"Hz with {nbursts} bursts"
    )


consistency_real, consistency_rsm = [
    {exp: [m.consistency for m in ms] for exp, (r, ms) in rs.items()}
    for rs in rasterses.values()
]


def separability(exp, X, pca=None, n_tries=100, validation=0.2):
    """
    Fit a linear classifier to the given features X and return its
    performance separating packet and non-packet units.
    """
    clf = SGDClassifier(n_jobs=12)
    if pca is not None and pca < X.shape[1]:
        clf = make_pipeline(PCA(n_components=pca), clf)
    y = ~(np.arange(X.shape[0]) < len(metricses[exp]["non_scaf_units"]))
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
    for exp, scoreses in consistency_real.items()
}

sep_on_fr = {exp: separability_on_fr(r) for exp, (r, _) in rasters_real.items()}


# %%

# The figure plots results for one example HMM first before comparing
# multiple, so pick a number of states and an experiment of interest.
# The figure compares three states of interest, which need to depend on the
# specific trained model we're looking at...
exp = "L1_t_spk_mat_sorted"
n_states = 15
r, models = rasters_real[exp]
model = models[np.nonzero(n_stateses == n_states)[0][0]]
interesting_states = [8, 9, 10]

# Compute hidden states throughout the recording, and use them to identify
# which states happen at which peak-relative times.
h = model.states(r)
lmargin_h, rmargin_h = model.burst_margins
peaks = r.find_bursts(margins=model.burst_margins)
state_prob = r.observed_state_probs(h, burst_margins=model.burst_margins)
state_order = r.state_order(h, model.burst_margins, n_states=n_states)
poprate = r.coarse_rate()
unit_order = np.int32(metricses[exp]["mean_rate_ordering"].flatten()) - 1
n_packet_units = len(metricses[exp]["scaf_units"])

# inverse_unit_order[i] is the index of unit i in unit_order.
inverse_unit_order = np.zeros_like(unit_order)
inverse_unit_order[unit_order] = np.arange(len(unit_order))

# Create a color map which is identical to gist_rainbow but with all the
# colors rescaled by 0.3 to make them match the ones above that are
# affected by alpha.
alpha_rainbow = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "alpha_rainbow", 0.5 + 0.5 * plt.get_cmap("gist_rainbow")(np.linspace(0, 1, 256))
)


with figure("Fig5", figsize=(8.5, 7.5)) as f:
    # Subfigure A: example burst rasters.
    A = f.subplots(
        1,
        3,
        gridspec_kw=dict(wspace=0.1, top=0.995, bottom=0.8, left=0.04, right=0.95),
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
        ax.set_xlim(*metricses[exp]["burst_window"].ravel() / 1e3)
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
        r"Non-Rigid \hspace{1.5em} Backbone", y=1.0, horizontalalignment="right"
    )

    # Subfigure B: state examples.
    BCtop, BCbot = 0.70, 0.41
    Bleft, Bwidth = 0.0, 0.65
    (Ba, RA), (Bb, RB), (Bc, RC) = [
        f.subplots(
            1,
            2,
            gridspec_kw=dict(
                top=BCtop,
                bottom=BCbot,
                width_ratios=[3, 1],
                wspace=0,
                left=Bleft + Bwidth * l,
                right=Bleft + Bwidth * r,
            ),
        )
        for l, r in [(0.06, 0.26), (0.4, 0.61), (0.76, 0.96)]
    ]
    deltas = dBA, dCB = [
        f.subplots(
            gridspec_kw=dict(
                top=BCtop,
                bottom=BCbot,
                left=Bleft + Bwidth * l,
                right=Bleft + Bwidth * r,
            )
        )
        for l, r in [(0.305, 0.365), (0.655, 0.715)]
    ]

    examples = [Ba, Bb, Bc]
    rates = [RA, RB, RC]
    for ax in examples:
        ax.set_xticks([])
        ax.set_xlabel("Realizations", rotation=25)
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
    Ba.set_ylabel(
        r"Non-Rigid \hspace{2.2cm} Backbone", y=0.99, horizontalalignment="right"
    )

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

    # Subfigure C: state heatmap.
    ax = f.subplots(gridspec_kw=dict(top=BCtop, bottom=BCbot, left=0.71, right=0.96))
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

    # Subfigure D: somehow show what's happening on the right.
    DEFtop, DEFbot = 0.33, 0.06
    D = f.subplots(
        1, 1, gridspec_kw=dict(top=DEFtop, bottom=DEFbot, left=0.03, right=0.3)
    )
    scores = consistency_real[exp][10]
    D.imshow(scores, aspect="auto", interpolation="none")
    D.set_xticks([])
    D.set_yticks([])
    D.set_xlabel(r"Non-Rigid \hspace{2.4cm} Backbone")
    D.set_ylabel("State")

    # Subfigure E: PCA of consistency scores for a single organoid, showing
    # that it's sufficient to separate packet/non-packet units.
    E = f.subplots(
        1, 1, gridspec_kw=dict(top=DEFtop, bottom=DEFbot, left=0.35, right=0.6)
    )
    scores = consistency_real[exp][10]
    is_packet = ~(np.arange(scores.shape[1]) < len(metricses[exp]["non_scaf_units"]))
    pca = PCA(n_components=2).fit_transform(scores.T)
    E.set_aspect("equal")
    E.scatter(pca[:, 1], pca[:, 0], c=is_packet)
    E.set_ylabel("PC1")
    E.set_xlabel("PC2")

    # Subfigure F: per-organoid separability metric.
    F = f.subplots(
        1, 1, gridspec_kw=dict(top=DEFtop, bottom=DEFbot, left=0.67, right=0.98)
    )
    F.violinplot(
        sep_on_states.values(),
        positions=np.arange(len(experiments)),
        showextrema=False,
        showmeans=True,
    )
    F.plot([], [], "C0_", ms=10, label="By State Structure")
    F.plot(sep_on_fr.values(), "_", ms=10, label="By Firing Rate")
    F.set_xlabel("Organoid")
    F.set_xticks(range(len(experiments)), [f"{i+1}" for i in range(len(experiments))])
    F.set_ylim([0.45, 1.05])
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    F.set_yticks(ticks, [f"{100*t:.0f}\\%" for t in ticks])
    F.set_ylabel("Backbone Classification Accuracy")
    F.legend(loc="lower right")
