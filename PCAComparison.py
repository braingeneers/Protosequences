# PCAComparison 
#
# Compare the variance explained by the first few principal components in
# each of the experiments under a given heading.
from tqdm.auto import tqdm
import numpy as np
import hmmsupport
from hmmsupport import get_raster, all_experiments, Model, figure
import matplotlib.pyplot as plt
import joblib

plt.ion()
hmmsupport.figdir('pca')

source = 'mouse'
bin_size_ms = 30
n_stateses = np.arange(10, 51)

experiments = all_experiments(source)
if source == 'organoid':
    experiments = [e for e in experiments
                   if e.startswith('L')]

def raster_valid(exp:str):
    try:
        r = get_raster(source, exp, bin_size_ms)
        return r.n_units > 15
    except Exception as e:
        print(exp, 'failed to load:', e)
        return False

rasters:dict[str,hmmsupport.Raster] = {
    exp: get_raster(source, exp, bin_size_ms)
    for exp in tqdm(experiments)
    if raster_valid(exp)}
experiments = list(rasters.keys())

print(len(experiments), 'experiments have any data')

bad_rasters:dict[str,hmmsupport.Raster] = {
    exp: get_raster(source, exp, bin_size_ms, 'rsm')
    for exp in tqdm(experiments)}

models:dict[str,list[Model]] = {
    exp: joblib.Parallel(n_jobs=16)(
        joblib.delayed(Model)(source, exp, bin_size_ms, n,
                              recompute_ok=False)
        for n in n_stateses)
    for exp in tqdm(experiments)}

bad_models:dict[str,list[Model]] = {
    exp: joblib.Parallel(n_jobs=16)(
        joblib.delayed(Model)(source, exp, bin_size_ms, n, 'rsm',
                              recompute_ok=False)
        for n in n_stateses)
    for exp in tqdm(experiments)}

are_ok = {k: np.logical_and(
    [m._hmm is not None for m in models[k]],
    [m._hmm is not None for m in bad_models[k]])
          for k in experiments}

models = {k: [m for m, ok in zip(models[k], are_ok[k]) if ok]
          for k in experiments}
bad_models = {k: [m for m, ok in zip(bad_models[k], are_ok[k]) if ok]
              for k in experiments}

# %%

from sklearn.decomposition import PCA

def stateses(exp:str, bad:bool):
    ms = (bad_models if bad else models)[exp]
    return [np.exp(m._hmm.observations.log_lambdas)
            for m in ms]

def raster(exp:str, bad:bool=False):
    return (bad_rasters if bad else rasters)[exp].raster

pcas = {exp: [PCA().fit(s)
              for s in stateses(exp, False)]
        for exp in experiments}
bad_pcas = {exp: [PCA().fit(s)
                  for s in stateses(exp, True)]
            for exp in experiments}

def variance_by_axis(pca:PCA, exp:str, bad:bool=False):
    return pca.explained_variance_[:10]
    return np.var(pca.transform(raster(exp, bad)), axis=0)[:10]

tev = {exp: np.array([variance_by_axis(pca, exp)
                      for pca in pcas[exp]])
       for exp in experiments}
pev = {e: tev[e] / tev[e].sum(axis=1, keepdims=True)
       for e in experiments}

bad_tev = {exp: np.array([variance_by_axis(pca, exp, bad=True)
                          for pca in bad_pcas[exp]])
           for exp in experiments}
bad_pev = {e: bad_tev[e] / bad_tev[e].sum(axis=1, keepdims=True)
           for e in experiments}

if source in ['organoid', 'eth']:
    for exp in tqdm(experiments):
        with figure(exp, save_exts=[]) as f:
            ax = f.gca()
            component = 1 + np.arange(10)
            ax.plot(component, pev[exp].T, 'C0')
            ax.plot(component, bad_pev[exp].T, 'C1')
            ax.set_ylabel('Percent Explained Variance')
            ax.set_xlabel('Principal Component')


# %%

def components_required(exp:str):
    return np.argmax(np.cumsum(pev[exp], axis=1)
                     >= bad_pev[exp][:,[0]], axis=1)

if source == 'organoid':
    with figure('Components Required') as f:
        ax = f.gca()
        ax.violinplot([components_required(exp)
                       for exp in experiments],
                      showmeans=True, showextrema=False)
        ax.set_xticks(np.arange(1,5),
                      [f'Organoid {i}' for i in range(1,5)])
        ax.set_ylabel('Components Required to Match Surrogate')

elif source == 'eth':
    with figure('Companents Required for ETH Organoids') as f:
        ax = f.gca()
        ax.violinplot([components_required(exp)
                       for exp in experiments],
                      showmeans=True, showextrema=False)
        ax.set_xticks(np.arange(1,5),
                      [f'Well {i}' for i in range(2,6)])

elif source == 'mouse':
    with figure('Mouse Dimensionality') as f:
        ax = f.gca()
        ax.hist([components_required(exp).mean()
                 for exp in experiments])
        ax.set_xlabel('Mean Components Required Across Models')
        ax.set_ylabel('Number of Experiments')


# %%

def transformed_data(exp:str, bad:bool):
    pca = (bad_pcas if bad else pcas)[exp][state_idx]
    return pca.transform(raster(exp, bad))[:,:3].T

def transformed_states(exp:str, bad:bool):
    pca = (bad_pcas if bad else pcas)[exp][state_idx]
    return pca.transform(stateses(exp, bad)[state_idx])[:,:3].T

state_idx = 0
if source == 'mouse':
    # These are the key experiments Mattia pointed out as having good
    # temporal structure.
    key_exps = ['1009-3', '1005-1', '366-2']
    with figure('Key Mouse Surrogate Comparison', figsize=(10,6)) as f:
        axes = f.subplots(2, 3, subplot_kw=dict(projection='3d'))
        for j, exp in enumerate(key_exps):
            axes[0,j].set_title(f'{exp}: {rasters[exp].n_units} units')
            for i, bad in enumerate([False, True]):
                axes[i,j].plot(*transformed_data(exp, bad),
                               color='grey', lw=0.1)
                axes[i,j].plot(*transformed_states(exp, bad), 'o')
