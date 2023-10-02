# Supplements.py
# Generate various miscellaneous supplemental figures.
import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from hmmsupport import figdir, figure

figdir("paper")
plt.ion()


# %%
# S20: The plateau that occurs above 10 states.

from cv_plateau_df import df as cv_plateau

with figure("Cross-Validation Plateau") as f:
    ax = sns.lineplot(
        data=cv_plateau,
        x="states",
        y="ll",
        hue="organoid",
        errorbar="sd",
    )
    ax.set_ylabel("Posterior Log Likelihood of True Data")
    ax.set_xlabel("Number of Hidden States in Model")
    ax.legend(title="Organoid")


# %%
# S14: Cross-validation proving that the model performance is better for
# the real data than for the shuffled data.

from cv_scores_df import df as cv_scores

with figure("Overall Model Validation") as f:
    ax = sns.boxplot(
        data=cv_scores,
        x="organoid",
        y="delta_ll",
        ax=f.gca(),
    )
    ax.set_ylabel("$\Delta$ Log Likelihood Real vs. Surrogate")
    ax.set_xlabel("Organoid")

# S21: Cross-validation by bin size.

with figure("Cross-Validation by Bin Size") as f:
    ax = f.gca()
    sns.violinplot(
        data=cv_scores,
        x="bin_size",
        y="train_ll",
        ax=ax,
    )
    ax.set_ylabel("Log Likelihood of True Data")
    ax.set_xlabel("Bin Size (ms)")

# %%
# S18: State traversal by model.

from state_traversal_df import df as traversed

with figure("States Traversed by Model") as f:
    ax = sns.violinplot(
        traversed,
        bw=0.1,
        x="model",
        y="rate",
        ax=f.gca(),
        cut=0,
        inner=None,
        scale="count",
    )
    ax.set_ylabel("Average States Traversed in Per Second in Scaffold Window")
    ax.set_xlabel("")
    ax.set_ylim(0, 40)

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
    ax.set_ylabel("Average States Traversed in Per Second in Scaffold Window")
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
