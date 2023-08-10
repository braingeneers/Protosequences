import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import stats

import hmmsupport


plt.ion()

source = "org_and_slice"
exps = hmmsupport.all_experiments(source)
mice = [e for e in exps if e[0] == "M"]
orgs = [e for e in exps if e[0] == "L"]
prims = [e for e in exps if e[0] == "P"]

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


def average_states_traversed(exp):
    peaks = srms[exp]["tburst"].ravel()
    start, stop = srms[exp]["scaf_window"].ravel()
    states_traversed = []
    for model in models[exp]:
        h = model.states(rasters[exp])
        states_traversed.append(0)

        def rd(x):
            return int(round(x / 30))

        for peak in peaks:
            states_traversed[-1] += len(set(h[rd(peak + start) : rd(peak + stop) + 1]))
        states_traversed[-1] /= len(peaks) * (stop - start)*1e-3

    return states_traversed


traversed = {
    "Mouse": np.hstack([average_states_traversed(exp) for exp in mice]),
    "Primary": np.hstack([average_states_traversed(exp) for exp in prims]),
    "Organoid": np.hstack([average_states_traversed(exp) for exp in orgs]),
}

plt.boxplot(traversed.values(), labels=traversed.keys())
plt.ylabel("Average States Traversed in Per Second in Scaffold Window")

print(stats.ks_2samp(traversed["Mouse"], traversed["Primary"]))
print(stats.ks_2samp(traversed["Mouse"], traversed["Organoid"]))
print(stats.ks_2samp(traversed["Organoid"], traversed["Primary"]))
