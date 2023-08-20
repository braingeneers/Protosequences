import math
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


def states_traversed(exp):
    """
    For each model for the given experiment, return the average number of
    distinct states traversed per second in the scaffold window.
    """
    start, stop = srms[exp]["scaf_window"].ravel()

    for model in models[exp]:
        T = model.bin_size_ms
        h = model.states(rasters[exp])
        length = math.ceil((stop - start) / T)
        yield [
            h[(bin0 := int((peak + start) / T)) : bin0 + length]
            for peak in srms[exp]["tburst"].ravel()
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

# %%


def entropy_of_states_traversed(exp):
    for model, stateses in zip(models[exp], states_traversed(exp)):
        dist = np.zeros((len(stateses[0]), model.n_states))
        for states in stateses:
            dist[np.arange(len(states)), states - 1] += 1 / len(stateses)
        yield stats.entropy(dist, axis=1)


def average_entropy(exp):
    start, stop = srms[exp]["scaf_window"].ravel()
    for entropies in entropy_of_states_traversed(exp):
        # Create weights which peak at zero and fall off to the left and
        # right.
        zero = round(len(entropies) * -start / (stop - start))
        weights = np.hstack(
            [
                np.linspace(0, 1, zero + 1, endpoint=False)[1:],
                np.linspace(1, 0, len(entropies) - zero, endpoint=False),
            ]
        )
        yield np.average(entropies, weights=weights)


entropies = {
    k: np.hstack([list(average_entropy(exp))
                  for exp in tqdm(exps, desc=k)])
    for k, exps in subsets.items()
}

plt.boxplot(entropies.values(), labels=traversed.keys())
plt.ylabel("Average Entropy of State Distribution Throughout Bursts")


for a, b in itertools.combinations(subsets.keys(), 2):
    ks = stats.ks_2samp(entropies[a], entropies[b])
    stat = ks.statistic
    if (p := ks.pvalue) < 1e-3:
        print(f"{a} vs. {b} is significant at ks = {stat:.2}, p = {100*p:.1e}%")
    else:
        print(f"{a} vs. {b} is insignificant (ks = {stat:.2}, {p = :.1%})")



# %%

def number_of_patterns(exp):
    '''
    Literally count how many distinct sequences of states are observed in each
    of the models for the given experiment.
    '''
    for model, stateses in zip(models[exp], states_traversed(exp)):
        yield len(set(tuple(states) for states in stateses))


counts = {
    k: np.hstack([list(number_of_patterns(exp))
                    for exp in tqdm(exps, desc=k)])
    for k, exps in subsets.items()
}


plt.boxplot(counts.values(), labels=counts.keys())
plt.ylabel("Number of Distinct Trajectories Observed")


for a, b in itertools.combinations(subsets.keys(), 2):
    ks = stats.ks_2samp(counts[a], counts[b])
    stat = ks.statistic
    if (p := ks.pvalue) < 1e-3:
        print(f"{a} vs. {b} is significant at ks = {stat:.2}, p = {100*p:.1e}%")
    else:
        print(f"{a} vs. {b} is insignificant (ks = {stat:.2}, {p = :.1%})")
