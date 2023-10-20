# Supplements.py
# Generate various miscellaneous supplemental figures.
import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from hmmsupport import (cv_plateau_df, cv_scores_df, figdir, figure,
                        state_traversal_df)

figdir("paper")
plt.ion()


# %%
# S20: The plateau that occurs above 10 states.

cv_plateau = cv_plateau_df()

with figure("Cross-Validation Plateau") as f:
    ax = sns.lineplot(
        data=cv_plateau,
        x="states",
        y="total_delta_ll",
        hue="organoid",
        errorbar="sd",
    )
    ax.set_ylabel("Total $\Delta$ Log Likelihood Real vs. Shuffled")
    ax.set_xlabel("Number of Hidden States in Model")
    ax.legend(title="Organoid")
    ax.set_yscale("log")


# %%
# S14: Cross-validation proving that the model performance is better for
# the real data than for the shuffled data.

cv_scores = cv_scores_df()

# This figure looks the same without the limitation of bin sizes to 30ms, but this
# version is easier to explain in the methods. :)
with figure("Overall Model Validation") as f:
    ax = sns.boxplot(
        data=cv_scores[cv_scores.bin_size == 30],
        x="organoid",
        y="total_delta_ll",
        ax=f.gca(),
    )
    ax.set_ylabel("Total $\Delta$ Log Likelihood Real vs. Shuffled")
    ax.set_xlabel("Organoid")
    ax.set_yscale("log")


# S21: Cross-validation by bin size.

with figure("Cross-Validation by Bin Size") as f:
    ax = f.gca()
    sns.boxplot(
        data=cv_scores,
        x="bin_size",
        y="total_delta_ll",
        ax=ax,
    )
    ax.set_ylabel("Total $\Delta$ Log Likelihood Real vs. Shuffled")
    ax.set_xlabel("Bin Size (ms)")
    ax.set_yscale("log")

# %%
# S18: State traversal by model.

traversed = state_traversal_df()

with figure("States Traversed by Model") as f:
    bins = np.arange(1, 38, 3)
    axes = f.subplots(3, 1)
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
        if i == 2:
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
