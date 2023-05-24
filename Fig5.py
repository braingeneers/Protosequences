#  Fig5.py
# Generate most of figure 5 of the final manuscript.
import os
import sys
import numpy as np
import scipy.io
from scipy import stats, signal, sparse, interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import hmmsupport
from hmmsupport import get_raster, figure, load_raw, Model
from sklearn.decomposition import PCA
import warnings
from tqdm import tqdm
import re


randomize = False
surr = 'rsm' if randomize else 'real'

if 'HMM_METHOD' in os.environ:
    hmm_library = os.environ['HMM_METHOD']
    figure_name = 'Fig5 ' + hmm_library
else:
    hmm_library = 'default'
    figure_name = 'Fig5'

if randomize:
    figure_name += ' Surrogate'

plt.ion()
hmmsupport.figdir('paper')

bin_size_ms = 30
n_states = 10, 20
n_stateses = np.arange(n_states[0], n_states[-1]+1)

source = 'organoid'
organoids_ages = {
    'L1_200123_2953_C': 7.7,
    'L2_200123_2950_C': 11.3,
    'L3_200123_2957_C': 11.3,
    'L5_200520_5116_C': 7.0,
}
experiments = list(organoids_ages.keys())


print('Loading fitted HMMs and calculating entropy.')
with tqdm(total=len(experiments)*len(n_stateses)) as pbar:
    rasters = {}
    for exp in experiments:
        rasters[exp] = get_raster(source, exp, bin_size_ms, surr), []
        for n in n_stateses:
            rasters[exp][1].append(Model(source, exp, bin_size_ms, n,
                                         surr, library=hmm_library))
            pbar.update()

# I can't read the Matlab file perfectly, so instead of TJ's experiment
# names, I have to hackily identify them by the number of units.
scaf_units = load_raw('scaf', 'scaf_unit_numbers')
scaf_unit_counts = [scaf_units['scaf_units'][0,i].shape[0]
                    + scaf_units['non_scaf_units'][0,i].shape[0]
                    for i in range(4)]
organoid_names = {e: re.search('_\d\d\d\d_', e).group(0)[1:-1]
                  for e in experiments}
scafs = {e: load_raw('scaf', f'{n}_cos_sim')['scaf_window'][0,:]/1e3
         for e, n in organoid_names.items()}

# Also read the burst edges and peaks from TJ's files for comparison to
# the ones I compute...
# Some of the arrays have mismatches, so I correct that manually too...
burst_edges = {e: load_raw('burstedges', f'{n}_BurstEdges')['edges'] * 1e3
               for e, n in organoid_names.items()}
burst_peaks = {e: load_raw('burstedges', f'{n}_BurstPeaks')['tburst'][0,:]
               for e, n in organoid_names.items()}
burst_peaks['L2_200123_2950_C'] = burst_peaks['L2_200123_2950_C'][1:]
burst_peaks['L3_200123_2957_C'] = burst_peaks['L3_200123_2957_C'][1:]

for k,(r,_) in rasters.items():
    nunits = r.raster.shape[1]
    meanfr = r.raster.mean() / r.bin_size_ms * 1000
    nbursts = len(r.find_bursts())
    print(f'{k} has {nunits} units firing at {meanfr:0.2f} '
          f'Hz with {nbursts} bursts')

good_experiments = [e for e,r in rasters.items()
                    if any(m is not None
                           for m in r[1])]

def good_models(exp):
    return [m for m in rasters[exp][1] if m is not None]

entropies = {e: [] for e in good_experiments}
baselines = {e: [] for e in good_experiments}
entropy_means, baseline_mean, baseline_std = {}, {}, {}
with tqdm(total=len(good_experiments)*len(n_stateses)) as pbar:
    for exp in good_experiments:
        for m in good_models(exp):
            m.compute_entropy(rasters[exp][0])
            pbar.update()
            entropies[exp].append(m.mean_entropy)
            baselines[exp].append(m.baseline_entropy)
        entropies[exp] = np.array(entropies[exp])
        entropy_means[exp] = entropies[exp].mean(axis=0)
        baseline_std[exp] = np.std(baselines[exp])
        baseline_mean[exp] = np.mean(baselines[exp])


# %%

n_states = 15
r = rasters[experiments[0]][0]
model = Model(source, experiments[0], bin_size_ms, n_states, surr,
              library=hmm_library)
model.compute_entropy(r)
h = model.states(r)

lmargin_h, rmargin_h = model.burst_margins
peaks = r.find_bursts(margins=model.burst_margins)
state_prob = r.observed_state_probs(h, burst_margins=model.burst_margins)
state_order = np.argsort(np.argmax(state_prob, axis=1))
poprate = r.coarse_rate()


with figure(figure_name, figsize=(8.5, 8.5)) as f:

    # Subfigure A: example burst rasters.
    axes = f.subplots(
        1, 3, gridspec_kw=dict(wspace=0.1,
                               top=0.995, bottom=0.82,
                               left=0.07, right=0.94))
    for ax, peak_float in zip(axes, peaks):
        peak = int(round(peak_float))
        when = slice(peak+lmargin_h, peak+rmargin_h+1)
        rsub = r.raster[when, :] / bin_size_ms
        hsub = np.array([np.nonzero(state_order == s)[0][0]
                         for s in h[when]])
        t_sec = (np.ogrid[when] - peak) * bin_size_ms / 1000
        ax.imshow(hsub.reshape((1,-1)), cmap='gist_rainbow',
                  aspect='auto', alpha=0.3, vmin=0, vmax=n_states-1,
                  extent=[t_sec[0], t_sec[-1], 0.5, rsub.shape[1]+0.5])
        idces, times_ms = r.spikes_within(when.start*bin_size_ms,
                                          when.stop*bin_size_ms)
        times = (times_ms - peak_float*bin_size_ms) / 1000
        ax.plot(times, idces+1, 'ko', markersize=0.5)
        ax.set_ylim(0.5, rsub.shape[1]+0.5)
        ax.set_xticks([0, 0.5])
        ax.set_xlim(-0.3, 0.6)
        ax.set_xlabel('Time from Peak (s)')
        ax.set_yticks([])
        ax2 = ax.twinx()
        when_ms = slice(when.start*bin_size_ms, when.stop*bin_size_ms)
        t_ms = np.ogrid[when_ms] - peak*bin_size_ms
        ax2.plot(t_ms/1e3, poprate[when_ms], 'r')
        ax2.set_ylim(0, 3)
        ax2.set_yticks([])

    ax2.set_ylabel('Population Rate (kHz)')
    ax2.set_yticks([0, 3])
    axes[0].set_ylabel('Neuron Unit ID')
    axes[0].set_yticks([1, rsub.shape[1]])

    # Subfigure B: state examples.
    BCtop, BCbot = 0.73, 0.5
    Bleft, Bwidth = 0.03, 0.6
    (A,RA), (B,RB), (C,RC) = [
        f.subplots(1, 2, gridspec_kw=dict(top=BCtop, bottom=BCbot,
                                          width_ratios=[3,1], wspace=0,
                                          left=Bleft+Bwidth*l,
                                          right=Bleft+Bwidth*r))
        for l,r in [(0.06, 0.26), (0.4, 0.61), (0.76, 0.96)]]
    deltas = dBA, dCB = [
        f.subplots(gridspec_kw=dict(top=BCtop, bottom=BCbot,
                                    left=Bleft+Bwidth*l,
                                    right=Bleft+Bwidth*r))
        for l,r in [(0.305,0.365), (0.655,0.715)]]

    examples = [A, B, C]
    rates = [RA, RB, RC]
    for ax in examples:
        ax.set_xticks([])
        # ax.set_xlabel('Realizations', rotation=35)
    for ax in rates:
        ax.set_xlim([0, 5])
        ax.set_xticks([0, 5])
        ax.set_xlabel('FR (Hz)')
    for ax in deltas:
        ax.set_xticks([-3, 2])
        ax.set_xlim([-3, 2])
        ax.set_xlabel('$\Delta$FR')
    for ax in examples + rates + deltas:
        ax.set_yticks([])
        ax.set_ylim(0.5, rsub.shape[1]+0.5)
    A.set_ylabel('Neuron Unit ID')
    A.set_yticks([1, rsub.shape[1]])

    states = [8, 9, 10]   # for the currently saved SSM model
    # states = [7, 8, 9]   # for the SSM model trained on surrogate data
    for axS, axH, s in zip(examples, rates, states):
        data = r.raster[h == state_order[s], :][:60, :]
        axS.set_title(f'State {s+1}')
        axS.imshow(data.T, aspect='auto', interpolation='none',
                   extent=[0, 1, r.raster.shape[1]+0.5, 0.5])

        axH.plot(data.mean(0), np.arange(r.raster.shape[1])+1,
                 c=plt.get_cmap('gist_rainbow')(s/(n_states-1)),
                 alpha=0.3)

    for ax, s0, s1 in zip(deltas, states[:-1], states[1:]):
        mu0 = r.raster[h == state_order[s0], :].mean(0)
        mu1 = r.raster[h == state_order[s1], :].mean(0)
        delta = mu1 - mu0
        ax.plot(delta, np.arange(r.raster.shape[1])+1,
                c='red', alpha=0.3)

    # Subfigure C: state heatmap.
    ax = f.subplots(gridspec_kw=dict(top=BCtop, bottom=BCbot,
                                     left=0.7, right=0.97))
    im = ax.imshow(state_prob[state_order, :], vmin=0, vmax=1,
                   extent=[t_sec[0], t_sec[-1],
                           n_states+0.5, 0.5],
                   interpolation='none', aspect='auto')
    ax.set_yticks([1, n_states])
    ax.set_xticks(0.3*np.arange(-1,3))
    ax.set_xlim(-0.3, 0.6)
    ax.set_xlabel('Time From Burst Peak (s)')
    ax.set_ylabel('Hidden State Number')
    plt.colorbar(im, ax=ax, label='Probability of Observing State',
                 aspect=10, ticks=[0, 1])

    # Subfigure D: entropy.
    axes = f.subplots(2, 4,
                      gridspec_kw=dict(hspace=0.1,
                                       height_ratios=[3,2],
                                       top=0.4, bottom=0.05,
                                       left=0.06, right=0.985))

    burst_widths = {e: (burst_edges[e]
                        - burst_peaks[e][:,None]).mean(0) / 1e3
                    for e in experiments}

    def hexcolor(r, g, b, a):
        r, g, b, a = [int(x*255) for x in (r, g, b, a)]
        return f'#{r:02x}{g:02x}{b:02x}{a:02x}'

    lmargin, rmargin = model.burst_margins

    en, pr = axes[0, :], axes[1, :]
    time_sec = np.arange(lmargin, rmargin+1) * bin_size_ms / 1000
    for i, exp in enumerate(good_experiments):
        # Plot the entropy on the left.
        ent = entropies[exp]
        c = f'C{i}'
        meanent = ent.mean(0)
        stdent = ent.std(0)
        minent = meanent - stdent
        maxent = meanent + stdent
        en[i].plot(time_sec, minent, c, label=exp)
        en[i].plot(time_sec, maxent, c)
        en[i].plot(time_sec, meanent, '--', c=c)
        en[i].fill_between(time_sec, minent, maxent,
                           alpha=0.5, color=c)

        for a in (en[i], pr[i]):
            a.axvspan(scafs[exp][0], scafs[exp][1],
                       color='gray', alpha=0.3)
            width = scafs[exp][1] - scafs[exp][0]
            a.set_xlim(burst_widths[exp])
            a.set_yticks([])
        en[i].set_xticks([])
        en[i].set_title(f'Organoid {i+1}')

        top = 4
        en[i].set_ylim(0, top)
        en[i].set_yticks([])
        # Plot the population rate for each burst on the right.
        r = rasters[exp][0]
        peaks = r.find_bursts(margins=model.burst_margins)
        poprate = r.coarse_rate()
        for peak in peaks:
            peak = int(peak * bin_size_ms)
            burst = poprate[peak+lmargin*bin_size_ms
                            :peak+(rmargin+1)*bin_size_ms]
            pr[i].plot(t_ms/1e3, burst, c, alpha=0.1)

    en[0].set_ylabel('Entropy (bits)')
    en[0].set_yticks([0, top])
    pr[0].set_ylabel('Normalized Pop. FR')
    for ax in pr:
        ax.set_xlabel('Time from Burst Peak (s)')
    f.align_ylabels(axes)
