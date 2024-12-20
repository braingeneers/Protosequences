# Supplements.py
# Generate supplementary figures 22, 23, 24, 26, and 34.
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from hmmsupport import (
    SHORT_NAME,
    cv_binsize_df,
    cv_plateau_df,
    figure,
    state_traversal_df,
)

# %%
# S22: Cross-validation by bin size.

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
# S23: The plateau that occurs above 10 states.

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


# S24: Cross-validation proving that the model performance is better for
# the real data than for the shuffled data.

subdata = cv_plateau[(cv_plateau.states >= 10) & (cv_plateau.states <= 30)]
melted = pd.melt(
    subdata, id_vars=["short_label"], value_vars=["total_ll", "total_surr_ll"]
)
melted.variable = melted.variable.map(
    dict(total_ll="Real", total_surr_ll="Shuffled").get
)

with figure("Overall Model Validation", figsize=(6.4, 6.4)) as f:
    raw, delta = f.subplots(2, 1)
    sns.boxplot(
        data=melted,
        x="short_label",
        hue="variable",
        y="value",
        width=1.0,
        ax=raw,
    )
    raw.set_ylabel("Total Log Likelihood")
    raw.legend()
    raw.set_xlabel("")
    sns.boxplot(data=subdata, x="short_label", y="total_delta_ll", ax=delta)
    delta.set_ylabel("Total $\\Delta$ Log Likelihood Real vs. Shuffled")
    delta.set_xlabel("")
    delta.set_yscale("log")
    f.align_ylabels()


# %%
# S26: State traversal by number of states.

traversal = state_traversal_df()
traversal.to_csv("traversal.csv")

with figure("States Traversed by K") as f:
    ax = sns.lineplot(
        traversal,
        x="K",
        y="rate",
        ax=f.gca(),
        hue="sample_type",
        errorbar="sd",
    )
    ax.set_ylabel("Average States Traversed in Per Second in Backbone Window")
    ax.set_xlabel("Number of Hidden States")
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(0, 30)
    ax.set_xticks([10, 15, 20, 25, 30])
    reg = stats.linregress(traversal.K, traversal.rate)
    ax.plot(
        [9.5, 30.5],
        reg.intercept + reg.slope * x,
        color="k",
        linestyle="--",
        label=f"Trendline ($r^2 = {reg.rvalue**2:.2}$)",
    )
    ax.legend(loc="lower right")

# S34: State traversal by model.

with figure("States Traversed by Model") as f:
    bins = np.arange(1, 38, 3)
    axes = f.subplots(4, 1)
    for (i, ax), (model, dfsub) in zip(
        enumerate(axes), traversal.groupby("sample_type")
    ):
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
