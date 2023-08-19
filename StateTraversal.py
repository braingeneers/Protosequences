import itertools

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import stats

import hmmsupport


plt.ion()

source = "org_and_slice"
exps = hmmsupport.all_experiments(source)
subsets = {
    "Mouse": [e for e in exps if e[0] == "M"],
    "Organoid": [e for e in exps if e[0] == "L"],
    "Primary": [e for e in exps if e[0] == "P"],
}

srms = {
    exp: hmmsupport.load_metrics(exp, error=False)
    for exp in tqdm(exps, desc="Loading metrics")
}

models = {
    exp: [
        hmmsupport.Model(source, exp, 30, K, recompute_ok=False) for K in range(10, 21)
    ]
    for exp in tqdm(exps, desc="Loading models")
}

rasters = {
    exp: hmmsupport.get_raster(source, exp, 30)
    for exp in tqdm(exps, desc="Loading rasters")
}

# %%


def list_of_states_traversed(exp):
    """
    For each model for the given experiment, return the average number of
    distinct states traversed per second in the scaffold window.
    """
    start, stop = srms[exp]["scaf_window"].ravel()

    def rd(x):
        return int(round(x / 30))

    for model in models[exp]:
        h = model.states(rasters[exp])
        yield [
            h[rd(peak + start) : rd(peak + stop) + 1]
            for peak in srms[exp]["tburst"].ravel()
        ]


def distinct_states_traversed(exp):
    """
    Calculate the average number of distinct states traversed per second
    in the scaffold window for each model for the provided experiment.
    """
    return [
        1e3 * np.mean([len(set(states)) / len(states) for states in model_states])
        for model_states in list_of_states_traversed(exp)
    ]


traversed = {
    k: np.hstack([distinct_states_traversed(exp) for exp in tqdm(exps, desc=k)])
    for k, exps in subsets.items()
}

plt.boxplot(traversed.values(), labels=traversed.keys())
plt.ylabel("Average States Traversed in Per Second in Scaffold Window")


for a, b in itertools.combinations(subsets.keys(), 2):
    ks = stats.ks_2samp(traversed[a], traversed[b])
    if (p := ks.pvalue) < 1e-3:
        stat = ks.statistic
        print(f"{a} vs. {b} is significant at ks = {stat:.2}, p = {100*p:.1e}% < 0.1%")
    else:
        print(f"{a} vs. {b} is insignificant ({p = :.1%})")
