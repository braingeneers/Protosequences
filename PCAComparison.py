from tqdm.auto import tqdm
import numpy as np
import hmmsupport
from hmmsupport import get_raster, all_experiments, Model, figure
import matplotlib.pyplot as plt
plt.ion()

source = 'organoid'
bin_size_ms = 30
n_states = 15

experiments = sorted([e for e in all_experiments(source) if
                      e.startswith('L')])
rasters:dict[str,hmmsupport.Raster] = {
    exp: get_raster(source, exp, bin_size_ms) for exp in tqdm(experiments)}
bad_rasters:dict[str,hmmsupport.Raster] = {
    exp: get_raster(source, exp, bin_size_ms, 'rsm') for exp in tqdm(experiments)}

models:dict[str,Model] = {
    exp: Model(source, exp, bin_size_ms, n_states)
    for exp in tqdm(experiments)}

bad_models:dict[str,Model] = {
    exp: Model(source, exp, bin_size_ms, n_states, 'rsm')
    for exp in tqdm(experiments)}


# %%

from sklearn.decomposition import PCA

def states(exp:str, bad:bool):
    m = (bad_models if bad else models)[exp]
    return np.exp(m._hmm.observations.log_lambdas)

pcas = {exp: PCA().fit(states(exp, False))
        for exp in tqdm(experiments)}
bad_pcas = {exp: PCA().fit(states(exp, True))
            for exp in tqdm(experiments)}

for exp in tqdm(experiments):
    with figure(exp, save_exts=[]) as f:
        ax = f.gca()
        ax.plot(state_pca(models[exp]).explained_variance_ratio_,
                label='Real Data')
        ax.plot(state_pca(bad_models[exp]).explained_variance_ratio_,
                label='Surrogate Data')
        ax.legend()
        ax.set_ylabel('Fraction of Variance Explained')
        ax.set_xlabel('Principal Component')

# %%

def transformed_data(exp:str, bad:bool):
    pca = (bad_pcas if bad else pcas)[exp]
    r = (bad_rasters if bad else rasters)[exp]
    return pca.transform(r.raster)[:,:2].T

def transformed_states(exp:str, bad:bool):
    pca = (bad_pcas if bad else pcas)[exp]
    return pca.transform(states(exp, bad))[:,:2].T

bad = False
for exp in tqdm(experiments):
    with figure(exp, save_exts=[]) as f:
        ax = f.gca()
        ax.plot(*transformed_data(exp, bad), alpha=0.5, color='grey')
        ax.plot(*transformed_states(exp, bad), 'o')
        # ax.plot(*transformed_data(exp, True)[:,:2].T)
