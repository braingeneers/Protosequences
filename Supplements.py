import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import hmmsupport
from hmmsupport import get_raster, Model, figure

hmmsupport.figdir("supplements")
plt.ion()


# %%
# Population Rate by State
# This supplement demonstrates that population rate isn't the only thing that
# distinguishes between states, because when they're ordered by burst location,
# they have a lot of overlap in population rate, but the overall sequence makes
# obvious sense.

source = "org_and_slice"
experiment = "L2_t_spk_mat_sorted"
bin_size_ms = 30
r = get_raster(source, experiment, bin_size_ms)

# Get state order from the model, same as Fig5.
n_states = 15
model = Model(source, experiment, bin_size_ms, n_states)
h = model.states(r)
burst_margins = -10, 20
peaks = r.find_bursts(margins=burst_margins)
state_prob = r.observed_state_probs(h, burst_margins=burst_margins)
state_order = r.state_order(h, burst_margins, n_states=n_states)
poprate_kHz = np.sum(r._raster, axis=1) / bin_size_ms

with figure("Population Rate by State") as f:
    poprate_by_state = [poprate_kHz[h == s] for s in state_order]
    ax = f.gca()
    ax.boxplot(poprate_by_state)
    ax.set_xticks(np.arange(1, n_states + 1))
    ax.set_ylabel("Population rate (kHz)")
    ax.set_xlabel("Hidden State")


# %%
# Cross-validation proving that the model performance is better for the real data than
# for the shuffled data.

source = "org_and_slice"
cv_score_files = glob.glob(f".cache/cv_scores/{source}_*")
df = []
for path in cv_score_files:
    # Load the scores from the file itself.
    with open(path, "rb") as f:
        scores = pickle.load(f)
        if "surrogate" not in scores:
            print(f"Skipping {path} because it doesn't have surrogate scores.")
            continue

    # Extract some metadata from the filename.
    name_parts = os.path.basename(path).split("_")
    organoid = name_parts[3]
    bin_size = name_parts[-3]
    states = name_parts[-2]

    # Combine those into dataframe rows, one per score rather than one per file
    # like a db normalization because plotting will expect that later anyway.
    df.extend(
        dict(
            organoid=organoid,
            bin_size=int(bin_size.removesuffix("ms")),
            states=int(states.removeprefix("K")),
            score=value,
        )
        for value in scores["validation"] - scores["surrogate"]
    )
df = pd.DataFrame(df)

with figure("Cross-Validation Scores") as f:
    ax = f.gca()
    sns.violinplot(
        data=df,
        x="organoid",
        y="score",
        ax=ax,
        inner=None,
        scale="count",
    )
    ax.set_ylabel("$\Delta$ Log Likelihood")
    ax.set_xlabel("Organoid")
