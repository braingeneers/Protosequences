# Supplements.py
# Generate various miscellaneous supplemental figures.
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

import hmmsupport
from hmmsupport import Model, all_experiments, cv_scores, figure, get_raster

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

source = "org_and_slice"
experiments = [e for e in all_experiments(source) if e.startswith("L")]
# Note that these *have* to be np.int64 because joblib uses argument hashes that
# are different for different integer types!
n_stateses = np.arange(10, 31)
bin_sizes_ms = np.array([10, 20, 30, 50, 70, 100])
cv_scoreses = {}
for p in tqdm(
    [
        (exp, bin_size_ms, n_states)
        for exp in experiments
        for bin_size_ms in bin_sizes_ms
        for n_states in n_stateses
    ]
):
    if cv_scores.check_call_in_cache(source, *p):
        cv_scoreses[p] = cv_scores(source, *p)
    else:
        tqdm.write(f"Missing {p}")

df = []
for (exp, bin_size_ms, num_states), scores in cv_scoreses.items():
    organoid = exp.split("_", 1)[0]

    # Combine those into dataframe rows, one per score rather than one per file
    # like a db normalization because plotting will expect that later anyway.
    df.extend(
        dict(
            organoid=organoid,
            bin_size=bin_size_ms,
            states=num_states,
            score=value,
        )
        for value in scores["validation"] - scores["surrogate"]
    )
df = pd.DataFrame(sorted(df, key=lambda row: int(row["organoid"][1:])))

with figure("Cross-Validation Scores") as f:
    ax = f.gca()
    sns.boxplot(
        data=df,
        x="organoid",
        y="score",
        ax=ax,
    )
    ax.set_ylabel("$\Delta$ Log Likelihood Real vs. Surrogate")
    ax.set_xlabel("Organoid")


# %%
# State traversal statistics.

source = "org_and_slice"
exps = hmmsupport.all_experiments(source)
n_stateses = range(10, 51)
subsets = {
    "Mouse": [e for e in exps if e[0] == "M"],
    "Organoid": [e for e in exps if e[0] == "L"],
    "Primary": [e for e in exps if e[0] == "P"],
}

metrics = {
    exp: hmmsupport.load_metrics(
        exp,
        only_include=["scaf_window", "tburst"],
        in_memory=False,
    )
    for exp in tqdm(exps, desc="Loading metrics")
}

models = {
    exp: [hmmsupport.Model(source, exp, 30, K) for K in n_stateses]
    for exp in tqdm(exps, desc="Loading models")
}

rasters = {
    exp: hmmsupport.get_raster(source, exp, 30)
    for exp in tqdm(exps, desc="Loading rasters")
}


def states_traversed(exp):
    """
    For each model for the given experiment, return the average number of
    distinct states traversed per second in the scaffold window.
    """
    start, stop = metrics[exp]["scaf_window"].ravel()

    for model in models[exp]:
        T = model.bin_size_ms
        h = model.states(rasters[exp])
        length = math.ceil((stop - start) / T)
        yield [
            h[(bin0 := int((peak + start) / T)) : bin0 + length]
            for peak in metrics[exp]["tburst"].ravel()
        ]


def distinct_states_traversed(exp):
    """
    Calculate the average number of distinct states traversed per second
    in the scaffold window for each model for the provided experiment.
    """
    return [
        1e3 * np.mean([len(set(states)) / len(states) for states in model_states])
        for model_states in states_traversed(exp)
    ]


traversed = pd.DataFrame(
    dict(
        traversed=count,
        model=model,
        exp=exp.split("_", 1)[0],
        n_states=n_states,
    )
    for model, exps in subsets.items()
    for exp in tqdm(exps, desc=model)
    for n_states, count in zip(n_stateses, distinct_states_traversed(exp))
)


with figure("States Traversed by Model") as f:
    groups = {k: vs.traversed for k, vs in traversed.groupby("model")}
    ax = sns.violinplot(
        traversed,
        x="model",
        y="traversed",
        ax=f.gca(),
        cut=0,
        inner=None,
        scale="count",
    )
    ax.set_ylabel("Average States Traversed in Per Second in Scaffold Window")
    ax.set_xlabel("")

with figure("States Traversed by Experiment") as f:
    ax = sns.violinplot(
        traversed,
        x="exp",
        y="traversed",
        ax=f.gca(),
        cut=0,
        inner=None,
        scale="count",
        hue="model",
    )
    ax.set_ylabel("Average States Traversed in Per Second in Scaffold Window")
    ax.set_xlabel("")
    ax.legend(loc="lower right")


for a, b in itertools.combinations(subsets.keys(), 2):
    ks = stats.ks_2samp(traversed[a], traversed[b])
    if (p := ks.pvalue) < 1e-3:
        stat = ks.statistic
        print(f"{a} vs. {b} is significant at ks = {stat:.2}, p = {100*p:.1e}% < 0.1%")
    else:
        print(f"{a} vs. {b} is insignificant ({p = :.1%})")
