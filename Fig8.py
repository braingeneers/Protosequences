# Fig8.py
# Generate my part of figure 8 of the final manuscript.
import os
import sys
import numpy as np
import scipy.io
from scipy import stats, signal, sparse, interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as plc
from hmmsupport import get_raster, figure, figdir, open, load_raw
from hmmsupport import cache_models, Model, all_experiments
from sklearn.decomposition import PCA
import glob
import warnings
from tqdm import tqdm
import re
import joblib


randomize = False
surr = 'rsm' if randomize else 'real'
age_subset = None

if 'HMM_METHOD' in os.environ:
    hmm_library = os.environ['HMM_METHOD']
    figure_name = 'Fig8 ' + hmm_library
else:
    hmm_library = 'default'
    figure_name = 'Fig8'

if randomize:
    figure_name += ' Surrogate'

plt.ion()
figdir('paper')

bin_size_ms = 30
n_states = 10, 50
n_stateses = np.arange(n_states[0], n_states[-1]+1)

source = 'mouse'
experiments = all_experiments(source)

if source == 'mouse':
    exp_age = {exp: load_raw(source, exp)['SUA'][0,0]['age'][0,0]
               for exp in experiments}
    def mouse_name(exp):
        return exp.split('-')[0]
elif source == 'adult-mouse':
    exp_age = {exp: int(exp[7:9]) for exp in experiments}
    def mouse_name(exp):
        return exp.split('[')[0][10:]

# Choose a subset of the experiments.
if age_subset is not None:
    experiments = [exp for exp in experiments
                   if exp_age[exp] in age_subset]

print('Fitting HMMs.')
cache_models(source, experiments, bin_size_ms, n_stateses, surr,
             library=hmm_library)

print('Loading fitted HMMs and calculating entropy.')
rasters = {
    exp: (get_raster(source, exp, bin_size_ms, surr),
          joblib.Parallel(n_jobs=24)(
              joblib.delayed(Model)(source, exp, bin_size_ms, n,
                                    library=hmm_library)
              for n in n_stateses))
    for exp in tqdm(experiments)}

mice = {}
for exp in rasters:
    mouse = mouse_name(exp)
    mice[mouse] = sorted(mice.get(mouse, []) + [exp],
                           key=lambda e: exp_age[e])
mice = {mouse: exps for mouse,exps in mice.items()
        if len({exp_age[e] for e in exps}) > 1}

good_experiments = [e for e,r in rasters.items()
                    if len(r[0].find_bursts()) > 30]

for k in good_experiments:
    r = rasters[k][0]
    nunits = r.raster.shape[1]
    totalfr = r.raster.mean() / r.bin_size_ms * nunits
    nbursts = len(r.find_bursts())
    print(f'{k} has {nunits} units firing at {totalfr:0.2f} '
          f'kHz total with {nbursts} bursts')

def good_models(exp):
    return [m for m in rasters[exp][1] if m is not None]

entropies, entropy_means, baselines, baseline_std = {}, {}, {}, {}
for exp in good_experiments:
    entropies[exp] = np.array([m.mean_entropy
                               for m in good_models(exp)])
    entropy_means[exp] = entropies[exp].mean(axis=0)
    baselines[exp] = np.mean([m.baseline_entropy
                              for m in good_models(exp)])
    baseline_std[exp] = np.std([m.baseline_entropy
                                for m in good_models(exp)])


# %%

n_states = 15
base_exp = '497-3'
r = rasters[base_exp][0]
model = Model(source, base_exp, bin_size_ms, n_states, library=hmm_library)
burst_margins = lmargin_h, rmargin_h = -10, 20
peaks = r.find_bursts(margins=burst_margins)
state_prob = r.observed_state_probs(model.h, burst_margins=burst_margins)
state_order = np.argsort(np.argmax(state_prob, axis=1))
lmargin, rmargin = model.burst_margins

with figure('Fig8', figsize=(8.5, 8.5)) as f:

    # Subfigure A: example burst rasters.
    idces, times_ms = np.nonzero(r.mat['spike_matrix'])
    axes = f.subplots(
        1, 3, gridspec_kw=dict(wspace=0.1,
                               top=0.995, bottom=0.82,
                               left=0.07, right=0.94))
    ax2s = [ax.twinx() for ax in axes]
    subpeaks = np.random.choice(peaks, 3)
    for ax, ax2, peak_float in zip(axes, ax2s, subpeaks):
        peak = int(round(peak_float))
        when = slice(peak+lmargin_h, peak+rmargin_h+1)
        rsub = r.raster[when, :] / bin_size_ms
        hsub = np.array([np.nonzero(state_order == s)[0][0] for s in model.h[when]])
        t_sec = (np.ogrid[when] - peak) * bin_size_ms / 1000
        ax.imshow(hsub.reshape((1,-1)), cmap='gist_rainbow',
                  aspect='auto', alpha=0.3, vmin=0, vmax=n_states-1,
                  extent=[t_sec[0], t_sec[-1], 0.5, rsub.shape[1]+0.5])
        idces, times = r.spikes_within(when.start*bin_size_ms,
                                       when.stop*bin_size_ms)
        times = (times - peak*bin_size_ms) / 1000
        ax.plot(times, idces+1, 'ko', markersize=0.5)
        ax.set_ylim(0.5, rsub.shape[1]+0.5)
        ax.set_xticks([0, 0.5])
        ax.set_xlim(t_sec[0], t_sec[-1])
        ax.set_xlabel('Time from Peak (s)')
        ax.set_yticks([])
        when_ms = slice(when.start*r.bin_size_ms, when.stop*r.bin_size_ms)
        peak_ms = peak * bin_size_ms
        t_ms = np.arange(-500, 1000)
        ax2.plot(t_ms / 1e3, r.coarse_rate()[t_ms + peak_ms], 'r')
        ax2.set_yticks([] if ax2 is not ax2s[-1]
                       else [0, ax2.get_yticks()[-1]])
    ax2.set_ylabel('Population Rate (kHz)')
    ymax = np.max([ax.get_ylim()[1] for ax in ax2s])
    for ax in ax2s:
        ax.set_ylim(0, ymax)

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
        ax.set_xlim([0, 0.5])
        ax.set_xticks([0, 0.5], ['$0$', '$0.5$'])
        ax.set_xlabel('FR (Hz)')
    for ax in deltas:
        ax.set_xticks([-0.3, 0], ['$-0.3$', '$0$'])
        ax.set_xlim([-0.5, 0.2])
        ax.set_xlabel('$\Delta$FR')
    for ax in examples + rates + deltas:
        ax.set_yticks([])
        ax.set_ylim(0.5, rsub.shape[1]+0.5)
    A.set_ylabel('Neuron Unit ID')
    A.set_yticks([1, rsub.shape[1]])

    states = np.subtract([8, 10, 11], 1)
    for axS, axH, s in zip(examples, rates, states):
        data = r.raster[model.h == state_order[s], :][:60, :]
        axS.set_title(f'State {s+1}')
        axS.imshow(data.T, aspect='auto', interpolation='none',
                   extent=[0, 1, r.raster.shape[1]+0.5, 0.5])

        axH.plot(data.mean(0), np.arange(r.raster.shape[1])+1,
                 c=plt.get_cmap('gist_rainbow')(s/(n_states-1)),
                 alpha=0.3)

    for ax, s0, s1 in zip(deltas, states[:-1], states[1:]):
        mu0 = r.raster[model.h == state_order[s0], :].mean(0)
        mu1 = r.raster[model.h == state_order[s1], :].mean(0)
        delta = mu1 - mu0
        ax.plot(delta, np.arange(r.raster.shape[1])+1,
                c='C3', alpha=0.3)

    # Subfigure C: state heatmap.
    axes[0].set_ylabel('Neuron Unit ID')
    axes[0].set_yticks([1, rsub.shape[1]])

    ax = f.subplots(gridspec_kw=dict(top=BCtop, bottom=BCbot,
                                     left=0.7, right=0.97))
    im = ax.imshow(state_prob[state_order, :], vmin=0, vmax=1,
                   extent=[t_sec[0], t_sec[-1],
                           n_states+0.5, 0.5],
                   aspect='auto')
    ax.set_yticks([1, n_states])
    ax.set_xticks(0.3*np.arange(-1,3))
    ax.set_xlabel('Time From Burst Peak (s)')
    ax.set_ylabel('Hidden State Number')
    plt.colorbar(im, ax=ax, label='Probability of Observing State',
                 aspect=10, ticks=[0, 1])

    # Subfigure D: entropy.
    en, pr = f.subplots(2, 1,
                        gridspec_kw=dict(hspace=0.1,
                                         height_ratios=[3,2],
                                         top=0.4, bottom=0.05,
                                         left=0.06, right=0.4))

    time_sec = np.arange(lmargin, rmargin+1) * bin_size_ms / 1000
    for exp in good_experiments:
        if exp == base_exp:
            continue
        ent = entropies[exp].mean(0)
        en.plot(time_sec, ent, '-', c=f'C0', alpha=0.5)
    ent = entropies[base_exp].mean(0)
    en.plot(time_sec, ent, '-', c=f'C3', lw=3)

    r = rasters[exp][0]
    peaks = r.find_bursts(margins=(lmargin, rmargin))
    for peak in peaks:
        peak_ms = int(round(peak * bin_size_ms))
        t_ms = np.arange(lmargin*bin_size_ms, rmargin*bin_size_ms+1)
        pr.plot(t_ms/1e3, r.coarse_rate()[peak_ms+t_ms[0]:peak_ms+t_ms[-1]+1],
                f'C3', alpha=0.2)

    top = 4
    en.set_ylim(0, top)
    en.set_yticks([])
    for a in (en, pr):
        # a.set_xlim(-0.2, 0.2)
        a.set_yticks([])
    en.set_xticks([])

    en.set_ylabel('Entropy (bits)')
    en.set_yticks([0, top])
    pr.set_ylabel('Normalized Pop. FR')
    pr.set_xlabel('Time from Burst Peak (s)')
    f.align_ylabels([en, pr])


# %%

for i, (exp,r) in enumerate(just_rasters.items()):
    if exp_age[exp] != 10:
        continue

    peaks = r.find_bursts()
    if len(peaks) < 30:
        print(f'Not enough bursts in {exp}')
        continue

    if r.data.sum(1).max() < 15:
        print(f'Not enough spikes in {exp}')
        continue

    plt.figure()
    for p in peaks:
        p_ms = int(round(p * bin_size_ms))
        plt.plot(r.fine_rate()[p_ms-200:p_ms+200])
        plt.title(exp)
