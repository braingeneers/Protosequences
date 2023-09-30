import math

import numpy as np
import pandas as pd

from hmmsupport import (Model, all_experiments, get_raster, load_metrics,
                        memoize)


@memoize
def state_traversal_df():
    source = "org_and_slice"
    exps = all_experiments(source)
    n_stateses = range(10, 51)
    subsets = {
        "Mouse": [e for e in exps if e[0] == "M"],
        "Organoid": [e for e in exps if e[0] == "L"],
        "Primary": [e for e in exps if e[0] == "P"],
    }

    only_include = ["scaf_window", "tburst"]
    metrics = {exp: load_metrics(exp, only_include) for exp in exps}
    print("Loaded metrics")

    models = {exp: [Model(source, exp, 30, K) for K in n_stateses] for exp in exps}
    print("Loaded models")

    rasters = {exp: get_raster(source, exp, 30) for exp in exps}
    print("Loaded rasters")

    def distinct_states_traversed(exp):
        """
        Calculate the average total number of distinct states as well
        as the rate at which they are traversed per second in the scaffold
        window for each model for the provided experiment.
        """
        start, stop = metrics[exp]["scaf_window"].ravel()
        length_ms = stop - start

        for model in models[exp]:
            T = model.bin_size_ms
            h = model.states(rasters[exp])
            length_bins = math.ceil(length_ms / T)
            state_seqs = [
                h[(bin0 := int((peak + start) / T)) : bin0 + length_bins]
                for peak in metrics[exp]["tburst"].ravel()
            ]
            distinct_states = [len(set(states)) for states in state_seqs]
            count = np.mean(distinct_states)
            rate = 1e3 * count / length_ms
            yield count, rate

    return pd.DataFrame(
        print(model, exp, n_states)
        or dict(
            count=count,
            rate=rate,
            model=model,
            exp=exp.split("_", 1)[0],
            n_states=n_states,
        )
        for model, exps in subsets.items()
        for exp in exps
        for n_states, (count, rate) in zip(n_stateses, distinct_states_traversed(exp))
    )


# This freaky boilerplate is necessary because joblib cache paths include the package
# name. It catches the case where the function is cached by running this file as a
# script: in that case, we manually import the packaged version of the function, and it
# gets cached under the packaged name.
if __name__ == "__main__":
    from state_traversal_df import df
else:
    df = state_traversal_df()
