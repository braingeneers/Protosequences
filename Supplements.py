import numpy as np
import matplotlib.pyplot as plt
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
model = Model(source, experiment, bin_size_ms, n_states, recompute_ok=False)
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
