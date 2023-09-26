# Supplements.py
# Generate various miscellaneous supplemental figures.
import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

import hmmsupport
from hmmsupport import Model, all_experiments, figure, get_raster

hmmsupport.figdir("paper")
plt.ion()


# %%
# Population Rate by State
# This supplement demonstrates that population rate isn't the only thing that
# distinguishes between states, because when they're ordered by burst location,
# they have a lot of overlap in population rate, but the overall sequence makes
# obvious sense.

with figure("Population Rate by State", figsize=(8.5, 11)) as f:
    source = "org_and_slice"
    bin_size_ms = 30
    exps = sorted(
        [e for e in all_experiments(source) if e.startswith("L")],
        key=lambda e: int(e.split("_", 1)[0][1:]),
    )
    axes = f.subplots(len(exps), 1)

    for experiment, ax in zip(exps, axes):
        organoid = int(experiment.split("_", 1)[0][1:])

        r = get_raster(source, experiment, bin_size_ms)

        # Get state order from the model, same as Fig5.
        n_states = 16
        model = Model(source, experiment, bin_size_ms, n_states)
        h = model.states(r)
        burst_margins = -10, 20
        peaks = r.find_bursts(margins=burst_margins)
        state_prob = r.observed_state_probs(h, burst_margins=burst_margins)
        state_order = r.state_order(h, burst_margins, n_states=n_states)
        poprate_kHz = np.sum(r._raster, axis=1) / bin_size_ms

        poprate_by_state = [poprate_kHz[h == s] for s in state_order]
        ax.boxplot(poprate_by_state)
        ax.set_ylabel(f"Organoid {organoid}\nPop.\\ rate (kHz)")
        if ax.get_ylim()[1] > 2:
            ax.set_yticks([0, 1, 2])
        else:
            ax.set_yticks([0, 1])
        ax.set_xticks(np.arange(1, n_states + 1))
        if experiment == exps[-1]:
            ax.set_xlabel("Hidden State")
        else:
            ax.set_xticklabels([])

    f.align_ylabels(axes)


# %%
# Cross-validation proving that the model performance is better for the real data than
# for the shuffled data.

from cv_scores_df import df as cv_scores

with figure("Overall Model Validation") as f:
    ax = f.gca()
    sns.boxplot(
        data=cv_scores,
        x="organoid",
        y="delta_ll",
        ax=ax,
    )
    ax.set_ylabel("$\Delta$ Log Likelihood Real vs. Surrogate")
    ax.set_xlabel("Organoid")

with figure("Cross-Validation by Bin Size") as f:
    ax = f.gca()
    sns.boxplot(
        data=cv_scores,
        x="bin_size",
        y="ll",
        ax=ax,
    )
    ax.set_ylabel("Log Likelihood of Model")
    ax.set_xlabel("Bin Size (ms)")

# %%
# State traversal statistics.

from state_traversal_df import df as traversed

with figure("States Traversed by Model") as f:
    groups = {k: vs.traversed for k, vs in traversed.groupby("model")}
    ax = sns.violinplot(
        traversed,
        bw=0.1,
        x="model",
        y="traversed",
        ax=f.gca(),
        cut=0,
        inner=None,
        scale="count",
    )
    ax.set_ylabel("Average States Traversed in Per Second in Scaffold Window")
    ax.set_xlabel("")

for a, b in itertools.combinations(groups.keys(), 2):
    ks = stats.ks_2samp(groups[a], groups[b])
    if (p := ks.pvalue) < 1e-3:
        stat = ks.statistic
        print(f"{a} vs. {b} is significant at ks = {stat:.2}, p = {100*p:.1e}% < 0.1%")
    else:
        print(f"{a} vs. {b} is insignificant ({p = :.2%})")
