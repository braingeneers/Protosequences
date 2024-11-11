# Supplements.py
# Generate various miscellaneous supplemental figures.
import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from hmmsupport import (
    SHORT_NAME,
    cv_plateau_df,
    cv_binsize_df,
    figure,
    state_traversal_df,
)

plt.ion()

# %%
# S15: The plateau that occurs above 10 states.

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


# S16: Cross-validation proving that the model performance is better for
# the real data than for the shuffled data.

with figure("Overall Model Validation") as f:
    ax = sns.boxplot(
        data=cv_plateau[cv_plateau.bin_size >= 10],
        x="short_label",
        hue="short_label",
        y="total_delta_ll",
        ax=f.gca(),
    )
    ax.set_ylabel("Total $\\Delta$ Log Likelihood Real vs. Shuffled")
    ax.set_yscale("log")
    ax.set_xlabel("")


# %%
# S17: Cross-validation by bin size.

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
# S26: State traversal by model.

traversed = state_traversal_df()
traversed["rate"] = traversed["count"] / traversed["burst_length_ms"] * 1000

with figure("States Traversed by Model") as f:
    bins = np.arange(1, 38, 3)
    axes = f.subplots(4, 1)
    for (i, ax), (model, dfsub) in zip(enumerate(axes), traversed.groupby("model")):
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


# S19: State traversal by number of states.

with figure("States Traversed by K") as f:
    ax = sns.lineplot(
        traversed,
        x="n_states",
        y="rate",
        ax=f.gca(),
        hue="model",
        errorbar="sd",
    )
    ax.set_ylabel("Average States Traversed in Per Second in Backbone Window")
    ax.set_xlabel("Number of Hidden States")
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(0, 40)
    reg = stats.linregress(traversed.n_states, traversed.rate)
    x = np.array([9, 51])
    ax.plot(
        x,
        reg.intercept + reg.slope * x,
        color="k",
        linestyle="--",
        label=f"Trendline ($r^2 = {reg.rvalue**2:.2}$)",
    )
    ax.legend(loc="lower right")


groups = {k: vs.rate for k, vs in traversed.groupby("model")}
for a, b in itertools.combinations(groups.keys(), 2):
    ks = stats.ks_2samp(groups[a], groups[b])
    if (p := ks.pvalue) < 1e-3:
        stat = ks.statistic
        print(f"{a} vs. {b} is significant at ks = {stat:.2}, p = {100*p:.1e}% < 0.1%")
    else:
        print(f"{a} vs. {b} is insignificant ({p = :.2%})")
