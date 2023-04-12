import os
import numpy as np
import scipy.io
from scipy import stats, signal, sparse, ndimage
from contextlib import contextmanager
import itertools
import functools
import pickle
import joblib
from tqdm.auto import tqdm
import re

try:
    from smart_open import open as _open
    import boto3
    boto3.setup_default_session(profile_name='nautilus')
    client = boto3.Session().client('s3', endpoint_url=os.environ.get(
        'AWS_ENDPOINT', 'https://s3-west.nrp-nautilus.io'))
    _open = functools.partial(_open, transport_params=dict(client=client))
except ImportError:
    from builtins import open as _open


DATA_DIR = ('./data' if os.path.isdir('data') else
            's3://braingeneers/personal/atspaeth/data')

S3_USER = os.environ.get('S3_USER')
CACHE_IS_LOCAL = S3_USER is None
CACHE_DIR = ('./.cache' if CACHE_IS_LOCAL else
             f's3://braingeneers/personal/{S3_USER}/cache')


class Cache:
    def __init__(self, func):
        self.wrapped = func
        functools.update_wrapper(self, func)

    @contextmanager
    def cache_file(self, mode, source, exp, bin_size_ms, n_states, surrogate):
        key = f'{source}_{exp}_{bin_size_ms}ms_{n_states}S_{surrogate}.pkl'
        path = os.path.join(CACHE_DIR, self.__name__, key)
        if CACHE_IS_LOCAL:
            directory = os.path.join(CACHE_DIR, self.__name__)
            if not os.path.isdir(directory):
                os.mkdir(directory)
        with _open(path, mode) as f:
            yield f

    def is_cached(self, source, exp, bin_size_ms, n_states, surrogate):
        try:
            with self.cache_file('rb', source, exp, bin_size_ms,
                                 n_states, surrogate) as f:
                return True
        except Exception:
            return False

    def get_cached(self, source, exp, bin_size_ms, n_states, surrogate):
        try:
            with self.cache_file('rb', source, exp, bin_size_ms,
                                 n_states, surrogate) as f:
                return pickle.load(f)
        except:
            return None

    def __call__(self, source, exp, bin_size_ms, n_states, surrogate, **kw):
        ret = self.get_cached(source, exp, bin_size_ms, n_states, surrogate)

        if ret is None:
            ret = self.wrapped(source, exp, bin_size_ms, n_states, surrogate, **kw)
            try:
                with self.cache_file('wb', source, exp, bin_size_ms,
                                     n_states, surrogate) as f:
                    pickle.dump(ret, f)
            except Exception as e:
                print('Failed to write to cache. :(', e)

        return ret


def figdir(path=None):
    if path is not None:
        path = os.path.expanduser(path.strip())
        if path == '':
            figdir.dir = 'figures'
        elif path[0] == '/':
            figdir.dir = path
        else:
            figdir.dir = os.path.join('figures', path)
        if not os.path.exists(figdir.dir):
            os.makedirs(figdir.dir)
    return os.path.abspath(figdir.dir)
figdir('')


def open(name, attr):
    return _open(os.path.join(figdir(), name), attr)


def experiment_parts(experiment):
    '''
    Separate the given experiment name into three parts: base name, file
    extension (following the last dot which is not the first character), and
    slice component (delimited by square brackets). Extension and slice are
    both optional.
    '''
    basename, extension, slice_text = re.match(
        r'^(.*?)(?:\.(\w+))?(?:\[(.*)\])?$', experiment).groups()

    if basename == '':
        basename, extension = extension, None

    return basename, extension, slice_text


FIT_ATOL = 1e-4
FIT_N_ITER = 1000

_HMM_METHODS = {}
try:
    from ssm import HMM as SSMHMM

    @Cache
    def _ssm_hmm(source, exp, bin_size_ms, n_states, surrogate,
                 verbose=False, atol=FIT_ATOL, n_iter=FIT_N_ITER):
        'Fit an HMM to data with SSM and return the model.'
        r = get_raster(bin_size_ms, source, exp, surrogate)
        hmm = SSMHMM(K=n_states, D=r.raster.shape[1],
                     observations='poisson')
        hmm.fit(r.raster, verbose=2 if verbose else 0,
                tolerance=atol, num_iters=n_iter)
        return hmm

    def _ssm_states(hmm, raster):
        'Return the most likely state sequence for the given raster.'
        return hmm.most_likely_states(raster)

    _HMM_METHODS['ssm'] = _ssm_hmm, _ssm_states
except ImportError:
    pass

try:
    import nonexistent
    from dynamax.hidden_markov_model import PoissonHMM as DynamaxHMM
    import jax.random as jr

    @Cache
    def _dynamax_hmm(source, exp, bin_size_ms, n_states, surrogate,
                     verbose=False, atol=FIT_ATOL, n_iter=FIT_N_ITER):
        'Fit an HMM to data with Dynamax and return the model.'
        r = get_raster(bin_size_ms, source, exp, surrogate)
        hmm = DynamaxHMM(num_states=n_states, emission_dim=r.raster.shape[1])
        hmm.params, props = hmm.initialize(jr.PRNGKey(np.random.randint(2**32)))
        hmm.params, lls = hmm.fit_em(hmm.params, props, r.raster,
                                     verbose=verbose, num_iters=n_iter)
        return hmm

    def _dynamax_states(hmm, raster):
        'Return the most likely state sequence for the given raster.'
        return hmm.most_likely_states(hmm.params, raster)

    _HMM_METHODS['dynamax'] = _dynamax_hmm, _dynamax_states
except ImportError:
    pass

try:
    from hmmlearn.hmm import PoissonHMM as HMMLearnHMM

    @Cache
    def _hmmlearn_hmm(source, exp, bin_size_ms, n_states, surrogate,
                      verbose=False, atol=FIT_ATOL, n_iter=FIT_N_ITER):
        'Fit an HMM to data with HMMLearn and return the model.'
        r = get_raster(bin_size_ms, source, exp, surrogate)
        hmm = HMMLearnHMM(n_components=n_states, verbose=False,
                          n_iter=n_iter, tol=atol)
        hmm.fit(r.raster)
        return hmm

    def _hmmlearn_states(hmm, raster):
        'Return the most likely state sequence for the given raster.'
        return hmm.predict(raster)

    _HMM_METHODS['hmmlearn'] = _hmmlearn_hmm, _hmmlearn_states
except ImportError:
    pass


try:
    import nonexistent
    from juliacall import Main as jl
    jl.seval('using Pkg; Pkg.activate("NeuroHMM"); using NeuroHMM')

    @Cache
    def _hmmbase_hmm(source, exp, bin_size_ms, n_states, surrogate,
                     verbose=False, atol=FIT_ATOL, n_iter=FIT_N_ITER):
        'Fit an HMM to data with NeuroHMM and return the model.'
        r = get_raster(bin_size_ms, source, exp, surrogate)
        display = jl.Symbol('iter' if verbose else 'none')
        hmm, _ = jl.NeuroHMM.fit_hmm(n_states, r.raster, tol=atol,
                                     maxiter=n_iter, display=display)
        return hmm

    def _hmmbase_states(hmm, raster):
        'Return the most likely state sequence for the given raster.'
        return np.asarray(jl.NeuroHMM.viterbi(hmm, raster)) - 1

    _HMM_METHODS['hmmbase'] = _hmmbase_hmm, _hmmbase_states
except ImportError:
    pass


# Set the default HMM fitting method to the first available.
if len(_HMM_METHODS) == 0:
    raise ImportError('No HMM libraries found.')
else:
    first_method = next(iter(_HMM_METHODS.keys()))
    default_method = os.environ.get('HMM_METHOD', first_method)
    if default_method not in _HMM_METHODS:
        raise ValueError(f'Invalid HMM_METHOD: {default_method}')
    _HMM_METHODS['default'] = _HMM_METHODS[default_method]


def get_fitted_hmm(source, exp, bin_size_ms, n_states, surrogate='real',
                   recompute_ok=False, library='default', verbose=False):
    if verbose:
        print(f'Running {source}/{exp}:{bin_size_ms}ms, K={n_states}')
    params = (np.str_(source), np.str_(exp), np.int64(bin_size_ms),
              np.int64(n_states), np.str_(surrogate))
    method = _HMM_METHODS[library][0]
    if recompute_ok or method.is_cached(*params):
        return method(*params, verbose=verbose)


class Model:
    def __init__(self, source, exp, bin_size_ms, n_states,
                 surrogate='real', lmargin_sec=-1.0, rmargin_sec=1.0,
                 library='default'):
        # Get all the (hopefully cached) heavy computations.
        self._hmm = get_fitted_hmm(source, exp, bin_size_ms, n_states,
                                   surrogate, library=library)
        self.r = get_raster(bin_size_ms, source, exp, surrogate)

        # Save metadata.
        self.source = source
        self.exp = exp
        self.bin_size_ms = bin_size_ms
        self.n_states = n_states
        self.surrogate = surrogate
        self.library = library

        # These are parameters for several other methods, where they need to
        # be in units of bins, not seconds.
        lmargin = int(lmargin_sec * 1000 / bin_size_ms)
        rmargin = int(rmargin_sec * 1000 / bin_size_ms)
        self.burst_margins = (lmargin, rmargin)

    def states(self, raster=None):
        return _HMM_METHODS[self.library][1](
            self._hmm, raster if raster is not None else self.r.raster)

    @property
    def burst_margins(self):
        '''
        The duration over which bursts are considered, expressed as a tuple
        of start and end time in bins relative to burst peak.
        '''
        return self._burst_margins

    @burst_margins.setter
    def burst_margins(self, value):
        # Actually set the margins.
        self._burst_margins = value

        # Recompute all the entropy measurements based on them.
        h = self.states()
        self.obs_state_prob = self.r.observed_state_probs(
            h, n_states=self.n_states,
            burst_margins=self.burst_margins)

        self.state_order = self.obs_state_prob.argmax(axis=1).argsort()

        overall_dist = np.zeros(self.n_states)
        for s in h:
            overall_dist[s] += 1/len(h)
        self.baseline_entropy = stats.entropy(overall_dist, base=2)

        self.mean_entropy = stats.entropy(self.obs_state_prob, axis=0, base=2)

    def dump(self, path):
        '''
        Dump the parameters of this HMM to a .mat file.
        '''
        mat = dict(
            state_means=np.exp(self._hmm.observations.log_lambdas
                               )[self._hmm.state_order,:],
            state_sequence=[np.nonzero(self._hmm.state_order == s)[0][0]
                            for s in self.states()],
            n_states=self.hmm.K,
            bin_size_ms=self.hmm.bin_size_ms)
        with open(path, 'wb') as f:
            scipy.io.savemat(f, mat)


def cache_models(source, experiments, bin_size_mses, n_stateses,
                 surrogates='real', library='default', verbose=False):
    '''
    Cache HMMs for all combinations of given parameters.

    Caching now happens in the fit method, so just shelve all the rasters
    once they're generated so we can collect them in the return. Most of
    the complexity now comes from getting a progress bar with a proper
    count of the total of models that need fitted.
    '''
    if verbose:
        print('Loading data from', DATA_DIR)
        print('Caching models computed by method', library, 'to', CACHE_DIR)

    argses = source, experiments, bin_size_mses, n_stateses, surrogates
    needs_run = []
    for p in itertools.product(*[np.atleast_1d(x) for x in argses]):
        if get_fitted_hmm(*p, library=library) is None:
            needs_run.append(p)

    if verbose:
        print(f'Need to re-run for {len(needs_run)} parameter values:')
        for p in needs_run:
            print('\t', *p)

    if not needs_run:
        return

    # Turn needs_run into a progress bar, but verbose breaks the bar, so
    # instead just let get_fitted_hmm print the parameters.
    for p in needs_run if verbose else tqdm(needs_run):
        try:
            get_fitted_hmm(*p, recompute_ok=True, library=library,
                           verbose=verbose)
        except ZeroDivisionError:
            s, e, bsms, n, surr = p
            surr = '' if surr == 'real' else f'({surr})'
            print(f'Failed to fit hmm to {s}:{e}{surr} with T={bsms}ms, K={n}.')


@contextmanager
def figure(name, save_args={}, save_exts=['png'], **kwargs):
    import matplotlib.pyplot as plt
    'Create a named figure and save it when done.'
    f = plt.figure(name, **kwargs)
    try:
        f.clf()
    except Exception:
        plt.close()
        f = plt.figure(name, **kwargs)

    yield f

    fname = name.lower().strip().replace(' ', '-')
    for ext in save_exts:
        if ext[0] != '.':
            ext = '.' + ext
        path = os.path.join(figdir(), fname + ext)
        f.savefig(path, **save_args)


def load_raw(source, experiment):
    'Load raw data from a file.'
    path = os.path.join(DATA_DIR, source,
                        experiment.removesuffix('.mat') + '.mat')
    with _open(path, 'rb') as f:
        return scipy.io.loadmat(f)


def exp_name_parts(exp):
    '''
    Split an experiment name into three parts, of which the latter two are
    optional: basename, file extension, and slice.

    Extension is the part of the name after the last dot, unless that would
    create an empty basename, in which case the basename is the whole name.
    Slice is not part of a filename; it is indicated in square brackets in
    Python syntax, and is used to select a subset of the data in seconds.
    '''
    exp, start, end = re.match(
        r'^(.*?)(?:\[([^\]]*):([^\]]*)\])?$', exp).groups()

    # Get rid of the extension.
    if '.' in exp[1:]:
        exp, _ = exp.rsplit('.', 1)

    # If there is a slice part, turn it into an actual slice in ms.
    if start is None:
        return exp, None
    try:
        conv = lambda x: int(float(x)*1e3) if x else None
        return exp, slice(conv(start), conv(end))
    except Exception:
        raise ValueError(
            'Invalid slice syntax in experiment name') from None


def _raster_poprate_from_units(length_ms, bin_size_ms, units):
    n_bins = int(np.ceil(length_ms / bin_size_ms))
    raster = np.zeros((n_bins, len(units)), int)
    poprate = np.zeros(int(length_ms))
    for i,unit in enumerate(units):
        for t in unit:
            raster[int(t // bin_size_ms), i] += 1
            poprate[int(t)] += 1
    return raster, poprate


def _raster_poprate_units_from_sm(length_ms, bin_size_ms, sm):
    sm = sparse.coo_array(sm)
    cols = sm.col // bin_size_ms
    n_bins = int(np.ceil(length_ms / bin_size_ms))
    raster = np.zeros((n_bins, sm.shape[0]), int)
    for d,r,c in zip(sm.data, sm.row, cols):
        raster[c,r] += d
    poprate = np.array(sm.sum(0)).ravel()
    idces, times = np.nonzero(sm)
    units = [times[idces == i] for i in range(sm.shape[0])]
    return raster, poprate, units



@functools.lru_cache
def get_raster(bin_size_ms, source, experiment, surrogate='real'):
    return Raster(bin_size_ms, source, experiment).get_surrogate(surrogate)


class Raster:
    surrogates: dict[str, type] = {}
    surrogate_name = 'Real Data'

    def __init__(self, bin_size_ms, source, experiment):
        self.bin_size_ms = bin_size_ms
        self.source = source
        self.experiment = experiment
        self.tag = f'{source}_{experiment}_{bin_size_ms}ms'
        experiment, sl = exp_name_parts(experiment)

        if sl:
            raise NotImplementedError('Slicing not implemented yet.')

        # First try loading the data from .mat files, in either Mattia's or
        # Tal's format...
        mat = load_raw(source, experiment)
        if 'SUA' in mat:
            mat['spike_matrix'] = mat['SUA'][0,0]['spike_matrix']

        # Mattia's data is formatted with something called a SUA, with
        # fixed sample rate of 1 kHz.
        if 'spike_matrix' in mat:
            sm = mat['spike_matrix']
            self.raster, self._poprate, self.units = \
                _raster_poprate_units_from_sm(sm.shape[1], bin_size_ms, sm)
            self.length_sec = sm.shape[1] / 1000

        # Tal and TJ's data is organized as units instead, and fs is stored.
        # The actual duration isn't stored, so just assume the recording was
        # a whole number of seconds long by rounding up the last spike time.
        else:
            self.units = [(unit[0][0]['spike_train']/mat['fs']*1e3)[0,:]
                          for unit in mat['units'][0]]
            self.length_sec = np.ceil(max(
                unit.max() for unit in self.units)/1e3)
            self.raster, self._poprate = _raster_poprate_from_units(
                1e3*self.length_sec, self.bin_size_ms, self.units)

        self.n_units = len(self.units)

        # If those .mat files don't exist, instead load from Sury's phy
        # zips. This can't work if the data is in a mat format, though.
        # except (OSError, FileNotFoundError):
        #     try:
        #         from braingeneers.analysis import read_phy_files
        #         self.sd = read_phy_files(
        #             os.path.join(DATA_DIR, source, experiment + '.zip'))
        #         if sl:
        #             self.sd = self.sd[sl]
        #     except AssertionError:
        #         raise FileNotFoundError(
        #             f'No data found for {source} {experiment}') from None
        #     self.data = self.sd.raster(bin_size_ms).T
        #     self.length_sec = self.sd.length / 1000

    def spikes_within(self, start_ms, end_ms):
        'Return unit indices and spike times within a time window.'
        unitsub = [t[(t >= start_ms) & (t < end_ms)]
                   for t in self.units]
        times = np.hstack(unitsub)
        idces = np.hstack([[i]*len(t) for i,t in enumerate(unitsub)])
        return idces, times

    def coarse_rate(self):
        return self.poprate(20, 100)

    def fine_rate(self):
        return self.poprate(5, 5)

    def poprate(self, square_width_ms, gaussian_width_ms=None):
        if gaussian_width_ms is None:
            gaussian_width_ms = square_width_ms

        # gaussian_filter1d uses the sigma of an IIR Gaussian, whereas TJ
        # uses a Matlab function that specifies the width of an FIR window
        # with sigma 1/5 of the width, so divide Gaussian width by 5.
        return ndimage.gaussian_filter1d(
            ndimage.uniform_filter1d(self._poprate*1.0, square_width_ms),
            gaussian_width_ms / 5)

    def get_surrogate(self, which):
        return Raster.surrogates[which](self)

    def average_burst_bounds_ms(self):
        "Average peak-relative start and end across all bursts."
        _, edges = self.find_burst_edges()
        return edges.mean(0)

    def find_burst_edges(self, use_rms=False):
        "Find the edges of the bursts in units of ms."
        # Find the peaks of the coarse rate.
        r_coarse = self.coarse_rate()
        peak_size_rms = 3 * np.sqrt(np.mean(r_coarse**2))
        peak_size_max = 0.5 * r_coarse.max()
        height = peak_size_rms if use_rms else peak_size_max
        peaks_ms = signal.find_peaks(r_coarse, height=height,
                                     distance=700)[0]

        # Descend from those peaks in both directions to find the first
        # points where the coarse rate is below 10% of the peak value.
        n = len(peaks_ms)
        edges = np.zeros((n, 2), int)
        for i, peak in enumerate(peaks_ms):
            min_height = 0.1 * r_coarse[peak]
            while (peak + edges[i,0] >= 0
                   and r_coarse[peak + edges[i,0]] > min_height):
                edges[i,0] -= 1
            while (peak + edges[i,1] < len(r_coarse)
                   and r_coarse[peak + edges[i,1]] > min_height):
                edges[i,1] += 1
        return peaks_ms, edges

    def find_bursts(self, margins=None, use_rms=False):
        '''
        Find the locations of burst peaks in units of bins, filtered to only
        those with the desired margins.
        '''
        r_fine = self.fine_rate()
        peaks, edges = self.find_burst_edges(use_rms=use_rms)
        peaks = np.array([
            peaks[i] + edges[i,0] + np.argmax(
                r_fine[peaks[i] + edges[i,0]:peaks[i] + edges[i,1]])
            for i in range(len(peaks))
        ]) / self.bin_size_ms

        # Filter out peaks too close to the edges of the recording.
        if margins is not None:
            try:
                lmargin, rmargin = margins
            except TypeError:
                lmargin, rmargin = -margins, margins
            peaks = peaks[(peaks + lmargin >= 0)
                          & (peaks + rmargin < self.raster.shape[0])]

        return peaks

    def observed_state_probs(self, h, burst_margins, n_states=None):
        '''
        Return a probability distribution of the states in h over time
        relative to each of this Raster's burst peaks. Automatically
        determines how many states to include, but can be overridden by
        passing n_states.
        '''
        if n_states is None:
            n_states = h.max()+1
        peaks = self.find_bursts(margins=burst_margins)
        lmargin, rmargin = burst_margins
        state_prob = np.zeros((n_states, rmargin-lmargin+1))

        for peak in peaks:
            peak_bin = int(round(peak))
            state_seq = h[peak_bin+lmargin:peak_bin+rmargin+1]
            for i, s in enumerate(state_seq):
                state_prob[s,i] += 1 / len(peaks)

        return state_prob


def surrogate(name=None):
    def wrap(subclass):
        lname = name or subclass.__name__
        if lname.startswith('_'):
            lname = lname[1:]

        Raster.surrogates[lname] = subclass
        return subclass

    return wrap

Raster.surrogates['real'] = lambda x: x


def _steal_metadata(dest, src, add_to_tag=None):
    dest.bin_size_ms = src.bin_size_ms
    dest.source = src.source
    dest.experiment = src.experiment
    dest.length_sec = src.length_sec
    dest.n_units = src.n_units
    if add_to_tag is None:
        dest.tag = src.tag
    else:
        dest.tag = src.tag + '_' + add_to_tag


@surrogate('rate')
class RateSurrogate(Raster):
    surrogate_name = 'Rate Surrogate'
    def __init__(self, raster):
        '''
        Surrogate raster formed by randomizing all the spike times uniformly
        within the total time interval. Preserves individual firing rates
        and nothing else.
        '''
        _steal_metadata(self, raster)
        self.units = [np.random.rand(len(t))*self.length_sec*1e3
                      for t in raster.units]
        self.raster, self._poprate = _raster_poprate_from_units(
            1e3*self.length_sec, self.bin_size_ms, self.units)
        assert np.all(self.raster.sum(0) == raster.raster.sum(0)),\
            'Failed to preserve individual rates.'


@surrogate('poprate')
class PoprateSurrogate(Raster):
    surrogate_name = 'Population Rate Surrogate'
    def __init__(self, raster):
        '''
        Surrogate raster formed by randomizing the unit ID for every spike.
        Preserves population rate and nothing else.
        '''
        _steal_metadata(self, raster)
        counts = np.cumsum([0] + [len(unit) for unit in raster.units])
        times = np.hstack(raster.units)
        np.random.shuffle(times)
        self.units = [times[a:b] for a,b in zip(counts, counts[1:])]
        self.raster, self._poprate = _raster_poprate_from_units(
            1e3*self.length_sec, self.bin_size_ms, self.units)
        assert np.all(self.raster.sum(1) == raster.raster.sum(1)),\
            'Failed to preserve population rates.'


def _spike_matrix_from_units(length_ms, units):
    'Create a spike matrix with 1ms bins from a list of spike trains.'
    indices = np.hstack(units).astype(int)
    indptr = np.cumsum([0] + [len(unit) for unit in units])
    data = np.ones_like(indices)
    return sparse.csr_array((data, indices, indptr),
                            (len(units), int(length_ms)))


@surrogate('tj')
class RandSpikeMatrix(Raster):
    surrogate_name = 'Randomized Spike Matrix'
    def __init__(self, raster, verbose=False):
        '''
        Return a raster which preserves the population rate and mean firing
        rate of each neuron by randomly reallocating all spike times to
        different neurons, using resampling to maintain the invariant that
        no neuron spikes twice in the same millisecond.

        To maximize the chances of selecting acceptable units without
        overlaps, I cancel out the 
        '''
        _steal_metadata(self, raster)
        sm = _spike_matrix_from_units(
            1e3*self.length_sec, raster.units)

        rsm = np.zeros(sm.shape, int)
        units = np.repeat(np.arange(self.n_units),
                          [len(t) for t in raster.units])
        n_spikeses = sm.sum(0)
        frame_order = np.argsort(n_spikeses)[::-1]
        for frame in tqdm(frame_order, disable=not verbose):
            n_spikes = n_spikeses[frame]
            if n_spikes > 0:
                for i in range(100):
                    rand_i = np.random.choice(len(units), n_spikes,
                                              replace=False)
                    rand_units = np.unique(units[rand_i])
                    if len(rand_units) == n_spikes:
                        unused_idces = np.isin(np.arange(len(units)),
                                               rand_i,
                                               assume_unique=True,
                                               invert=True)
                        units = units[unused_idces]
                        break
                else:
                    print('Dropping spikes due to statistics. :(')
                rsm[rand_units,frame] = 1

        self.raster, self._poprate, self.units = \
            _raster_poprate_units_from_sm(1e3*self.length_sec,
                                          self.bin_size_ms, rsm)

@surrogate('rsm')
class RandSpikeMatrix2(Raster):
    surrogate_name = 'Randomized Spike Matrix'
    def __init__(self, raster):
        _steal_metadata(self, raster)
        sm = _spike_matrix_from_units(
            1e3*self.length_sec, raster.units)

        rsm = np.zeros(sm.shape, int)
        units = np.arange(self.n_units)
        weights = np.array([len(t) for t in raster.units])

        # Iterate over the frames in order of how many spikes they have,
        # skipping all the frames with none.
        n_spikeses = sm.sum(0)
        frame_order = np.argsort(n_spikeses)[::-1]
        frame_order = frame_order[n_spikeses[frame_order] != 0]

        for frame in frame_order:
            n_spikes = n_spikeses[frame]
            p = weights/weights.sum()
            rand_units = np.random.choice(units, n_spikes,
                                          replace=False, p=p)
            weights[rand_units] -= 1
            rsm[rand_units,frame] = 1

        self.raster, self._poprate, self.units = \
            _raster_poprate_units_from_sm(1e3*self.length_sec,
                                          self.bin_size_ms, rsm)
