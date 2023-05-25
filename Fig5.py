#  Fig5.py
# Generate most of figure 5 of the final manuscript.
import numpy as np
import matplotlib.pyplot as plt
import hmmsupport
from hmmsupport import get_raster, figure, load_raw, Model, all_experiments
from sklearn.decomposition import PCA
from tqdm import tqdm


source = 'organoid'
experiments = [x for x in all_experiments(source)
               if source != 'organoid' or x.startswith('L')]

surr = 'real'
hmm_library = 'default'

figure_name = f'Fig5 {source}'
if hmm_library != 'default':
    figure_name += ' ' + hmm_library
if surr != 'real':
    figure_name += ' Surrogate'

plt.ion()
hmmsupport.figdir('paper')

bin_size_ms = 30
n_states = 10, 20
n_stateses = np.arange(n_states[0], n_states[-1]+1)

print('Loading fitted HMMs and calculating entropy.')
srms = {}
with tqdm(total=len(experiments)*(1+len(n_stateses))) as pbar:
    rasters = {}
    for exp in experiments:
        srms[exp] = load_raw(
            'metrics', exp.split('_')[0] + '_single_recording_metrics')
        rasters[exp] = get_raster(source, exp, bin_size_ms, surr), []
        pbar.update()
        window = srms[exp]['burst_window'][0,:]/1e3
        for n in n_stateses:
            rasters[exp][1].append(Model(source, exp, bin_size_ms, n,
                                         surr, library=hmm_library))
            rasters[exp][1][-1].compute_entropy(rasters[exp][0], *window)
            pbar.update()

for k,(r,_) in rasters.items():
    nunits = r.raster.shape[1]
    meanfr = r.raster.mean() / r.bin_size_ms * 1000
    nbursts = len(r.find_bursts())
    print(f'{k} has {nunits} units firing at {meanfr:0.2f} '
          f'Hz with {nbursts} bursts')


# %%

# The figure plots results for one example HMM first before comparing
# multiple, so pick a number of states and an experiment of interest.
n_states = 15
r,models = rasters[experiments[0]]
model = models[np.nonzero(n_stateses == n_states)[0][0]]

# Compute hidden states throughout the recording, and use them to identify
# which states happen at which peak-relative times.
h = model.states(r)
lmargin_h, rmargin_h = model.burst_margins
peaks = r.find_bursts(margins=model.burst_margins)
state_prob = r.observed_state_probs(h, burst_margins=model.burst_margins)
state_order = np.argsort(np.argmax(state_prob, axis=1))
poprate = r.coarse_rate()
# unit_order = srms[experiments[0]]['mean_rate_ordering'][0,:] - 1
unit_order = np.vstack([srms[experiments[0]]['scaf_units'],
                        srms[experiments[0]]['non_scaf_units']])[:,0] - 1
inverse_unit_order = np.argsort(unit_order)

# The figure compares three states of interest, which need to depend on the
# specific trained model we're looking at...
match source, hmm_library, randomize, n_states:
    case 'organoid', 'default', False, 15:
        interesting_states = [8, 9, 10]
    case 'organoid', 'default', True, 15:
        interesting_states = [7, 8, 9]
    case 'eth', 'default', False, 15:
        interesting_states = [4, 5, 6]
    case _:
        print('No interesting states chosen yet for these parameters.')
        interesting_states = state_prob[state_order,:].max(1).argsort()[-3:]

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
        ax.plot(times, inverse_unit_order[idces]+1, 'ko', markersize=0.5)
        ax.set_ylim(0.5, rsub.shape[1]+0.5)
        ax.set_xticks([0, 0.5])
        # ax.set_xlim(-0.3, 0.6)
        ax.set_xlim(*srms[experiments[0]]['burst_window'][0,:]/1e3)
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
        ax.set_xlabel('Realizations', rotation=35)
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

    for axS, axH, s in zip(examples, rates, interesting_states):
        data = r.raster[h == state_order[s], :][:, unit_order]
        data_sub = data[np.random.choice(data.shape[0], 60),:]
        axS.set_title(f'State {s+1}')
        axS.imshow(data_sub.T, aspect='auto', interpolation='none',
                   vmin=0, vmax=r.raster.max(),
                   extent=[0, 1, r.raster.shape[1]+0.5, 0.5])

        axH.plot(data.mean(0), np.arange(r.raster.shape[1])+1,
                 c=plt.get_cmap('gist_rainbow')(s/(n_states-1)),
                 alpha=0.3)

    for ax, s0, s1 in zip(deltas, interesting_states[:-1],
                          interesting_states[1:]):
        mu0 = r.raster[h == state_order[s0], :].mean(0)
        mu1 = r.raster[h == state_order[s1], :].mean(0)
        delta = mu1 - mu0
        ax.plot(delta[unit_order], np.arange(r.raster.shape[1])+1,
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

    def hexcolor(r, g, b, a):
        r, g, b, a = [int(x*255) for x in (r, g, b, a)]
        return f'#{r:02x}{g:02x}{b:02x}{a:02x}'

    en, pr = axes[0, :], axes[1, :]
    for i, exp in enumerate(experiments):
        lmargin, rmargin = rasters[exp][1][0].burst_margins
        time_sec = np.arange(lmargin, rmargin+1) * bin_size_ms / 1000
        ent = np.array([m.mean_entropy for m in rasters[exp][1]])
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
            a.axvspan(*srms[exp]['scaf_window'][0,:]/1e3,
                      color='gray', alpha=0.3)
            a.set_xlim(srms[exp]['burst_window'][0,:]/1e3)
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
        t_ms = np.arange(lmargin*bin_size_ms, (rmargin+1)*bin_size_ms)
        for peak in peaks:
            peak_ms = int(round(peak * bin_size_ms))
            burst = poprate[peak_ms+lmargin*bin_size_ms
                            :peak_ms+(rmargin+1)*bin_size_ms]
            pr[i].plot(t_ms/1e3, burst, c, alpha=0.1)

    en[0].set_ylabel('Entropy (bits)')
    en[0].set_yticks([0, top])
    pr[0].set_ylabel('Normalized Pop. FR')
    for ax in pr:
        ax.set_xlabel('Time from Burst Peak (s)')
    f.align_ylabels(axes)
