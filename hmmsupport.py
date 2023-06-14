import itertools
import functools
import re
import glob
import os
import sys
import pickle
import joblib
import h5py
import mat73
import numpy as np
import scipy.io
import awswrangler
import tempfile
from io import BytesIO
from tqdm.auto import tqdm
from contextlib import contextmanager
from scipy import stats, signal, sparse, ndimage
from braingeneers.analysis import read_phy_files
from braingeneers.utils.smart_open_braingeneers import open


def s3_isdir(path):
    try:
        next(awswrangler.s3.list_objects(path, chunked=True))
        return True
    except StopIteration:
        return False


DATA_DIR = 'data'


@functools.lru_cache
def data_dir(source):
    path = os.path.join(DATA_DIR, source)
    if os.path.isdir(path):
        return path
    elif path.startswith('s3://') and s3_isdir(path):
        return path

    personal_s3 = 's3://braingeneers/personal/atspaeth/data/' + source
    if s3_isdir(personal_s3):
        return personal_s3

    main_s3 = f's3://braingeneers/ephys/{source}/derived/kilosort2'
    if s3_isdir(main_s3):
        return main_s3

    raise FileNotFoundError(f'Could not find data for {source}')


def all_experiments(source):
    path = data_dir(source)
    if path.startswith('s3://'):
        paths = awswrangler.s3.list_objects(path + '/')
    else:
        paths = glob.glob(os.path.join(path, '*'))
    return sorted([
        os.path.splitext(os.path.basename(x))[0]
        for x in paths])


CACHE_DIR = '.cache'
S3_USER = os.environ.get('S3_USER')
S3_CACHE = f's3://braingeneers/personal/{S3_USER}/cache'
CACHE_ROOTS = [root_path for (root_path, valid) in [
    (CACHE_DIR, os.path.isdir(CACHE_DIR)),
    (S3_CACHE, S3_USER is not None)
] if valid]


def _store(obj, path):
    '''
    Pickle an object to a path.

    If the path is local, ensure that it exists before trying to open the
    file.
    '''
    iss3 = path.startswith('s3://')
    try:
        if not iss3:
            dirname = os.path.dirname(path)
            os.makedirs(dirname, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        verb = 'upload' if iss3 else 'save'
        print(f'Failed to {verb} {path}: {e}', file=sys.stderr)


class Cache:
    def __init__(self, func):
        self.wrapped = func
        functools.update_wrapper(self, func)

    def _cache_names(self, source, exp, bin_size_ms, n_states, surrogate):
        'Local cache file and S3 object cache path.'
        key = f'{source}_{exp}_{bin_size_ms}ms_{n_states}S_{surrogate}.pkl'
        file = os.path.join(CACHE_DIR, self.__name__, key)
        s3_object = '/'.join((S3_CACHE, self.__name__, key))
        return file, s3_object

    def _cache_path(self, *args):
        'The nearest possible cache path to read from, None if uncached.'
        filename, s3_object = self._cache_names(*args)

        if os.path.isdir(CACHE_DIR) and os.path.isfile(filename):
            return filename
        elif S3_USER and awswrangler.s3.does_object_exist(s3_object):
            return s3_object
        else:
            return None

    def is_cached(self, *args):
        'Whether the given parameters are cached.'
        return self._cache_path(*args) is not None

    def get_cached(self, *args):
        '''
        Get the cached object, or None if unavailable.

        Additionally, if restoring the object from S3 when a local cache is
        present, add the object to the local cache.
        '''
        path = self._cache_path(*args)

        if path is None:
            return None

        with open(path, 'rb') as f:
            ret = pickle.load(f)

        if path.startswith('s3://') and os.path.isdir(CACHE_DIR):
            _store(ret, self._cache_names(*args)[0])

        return ret

    def __call__(self, *args, **kw):
        ret = self.get_cached(*args)

        if ret is None:
            ret = self.wrapped(*args, **kw)

            filename, s3_object = self._cache_names(*args)

            if os.path.isdir(CACHE_DIR):
                _store(ret, filename)

            if S3_USER is not None:
                _store(ret, s3_object)

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


FIT_ATOL = 1e-3
FIT_N_ITER = 5000

_HMM_METHODS = {}
class HMMMethod:
    def __init__(self, library, fit, states):
        self.library = library
        self.fit = fit
        self.states = states
        _HMM_METHODS[library] = self

try:
    from ssm import HMM as SSMHMM

    @Cache
    def _ssm_hmm(source, exp, bin_size_ms, n_states, surrogate,
                 verbose=False, atol=FIT_ATOL, n_iter=FIT_N_ITER):
        'Fit an HMM to data with SSM and return the model.'
        r = get_raster(source, exp, bin_size_ms, surrogate)
        hmm = SSMHMM(K=n_states, D=r.raster.shape[1],
                     observations='poisson')
        hmm.fit(r.raster, verbose=2 if verbose else 0,
                tolerance=atol, num_iters=n_iter)
        return hmm

    def _ssm_states(hmm, raster):
        'Return the most likely state sequence for the given raster.'
        return hmm.most_likely_states(raster)

    SSMFit = HMMMethod('ssm', _ssm_hmm, _ssm_states)

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
        r = get_raster(source, exp, bin_size_ms, surrogate)
        hmm = DynamaxHMM(num_states=n_states, emission_dim=r.raster.shape[1])
        hmm.params, props = hmm.initialize(jr.PRNGKey(np.random.randint(2**32)))
        hmm.params, lls = hmm.fit_em(hmm.params, props, r.raster,
                                     verbose=verbose, num_iters=n_iter)
        return hmm

    def _dynamax_states(hmm, raster):
        'Return the most likely state sequence for the given raster.'
        return hmm.most_likely_states(hmm.params, raster)

    DynamaxFit = HMMMethod('dynamax', _dynamax_hmm, _dynamax_states)

except ImportError:
    pass

try:
    from hmmlearn.hmm import PoissonHMM as HMMLearnHMM

    @Cache
    def _hmmlearn_hmm(source, exp, bin_size_ms, n_states, surrogate,
                      verbose=False, atol=FIT_ATOL, n_iter=FIT_N_ITER):
        'Fit an HMM to data with HMMLearn and return the model.'
        r = get_raster(source, exp, bin_size_ms, surrogate)
        hmm = HMMLearnHMM(n_components=n_states, verbose=False,
                          n_iter=n_iter, tol=atol)
        hmm.fit(r.raster)
        return hmm

    def _hmmlearn_states(hmm, raster):
        'Return the most likely state sequence for the given raster.'
        return hmm.predict(raster)

    HMMLearnFit = HMMMethod('hmmlearn', _hmmlearn_hmm, _hmmlearn_states)

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
        r = get_raster(source, exp, bin_size_ms, surrogate)
        display = jl.Symbol('iter' if verbose else 'none')
        hmm, _ = jl.NeuroHMM.fit_hmm(n_states, r.raster, tol=atol,
                                     maxiter=n_iter, display=display)
        return hmm

    def _hmmbase_states(hmm, raster):
        'Return the most likely state sequence for the given raster.'
        return np.asarray(jl.NeuroHMM.viterbi(hmm, raster)) - 1

    HMMBaseFit = HMMMethod('hmmbase', _hmmbase_hmm, _hmmbase_states)

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


def is_cached(source, exp, bin_size_ms, n_states, surrogate='real',
              library='default'):
    'Return whether the given model is cached.'
    return _HMM_METHODS[library].fit.is_cached(source, exp, bin_size_ms,
                                               n_states, surrogate)


def get_fitted_hmm(source, exp, bin_size_ms, n_states, surrogate='real',
                   recompute_ok=False, library='default', verbose=False):
    if verbose:
        print(f'Running {source}/{exp}:{bin_size_ms}ms, K={n_states}')
    params = source, exp, bin_size_ms, n_states, surrogate
    method = _HMM_METHODS[library]
    if recompute_ok:
        return method.fit(*params, verbose=verbose)
    else:
        return method.fit.get_cached(*params)


class Model:
    def __init__(self, source, exp, bin_size_ms, n_states,
                 surrogate='real', library='default',
                 verbose=False, recompute_ok=True):

        # Retrieve the (hopefully cached) model.
        self._hmm = get_fitted_hmm(source, exp, bin_size_ms, n_states,
                                   surrogate, library=library,
                                   verbose=verbose,
                                   recompute_ok=recompute_ok)

        # Save metadata.
        self.source = source
        self.exp = exp
        self.bin_size_ms = bin_size_ms
        self.n_states = n_states
        self.surrogate = surrogate
        self.library = library

    def states(self, raster):
        return _HMM_METHODS[self.library].states(self._hmm, raster.raster)

    def compute_entropy(self, raster=None, lmargin_sec=-1.0, rmargin_sec=1.0):
        # Load the raster if not provided. You shouldn't actually need to
        # provide it, as it is cached in the kernel and there's no reason
        # to compute entropy on the wrong data.
        if raster is None:
            raster = get_raster(self.source, self.exp, self.bin_size_ms,
                                self.surrogate)

        # Save the burst margins in units of bins.
        lmargin = int(lmargin_sec * 1000 / self.bin_size_ms)
        rmargin = int(rmargin_sec * 1000 / self.bin_size_ms)
        self.burst_margins = (lmargin, rmargin)

        # Recompute all the entropy measurements based on them.
        h = self.states(raster)
        self.obs_state_prob = raster.observed_state_probs(
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
        # This will be a regular file, not S3, so don't bother with the
        # BytesIO workaround.
        with open(os.path.join(figdir(), path), 'wb') as f:
            scipy.io.savemat(f, mat)


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


def _load73(filename, only_include=None):
    '''
    Load raw data from a Matlab 7.3 file on disk, only including the
    specified variables. Silently ignore missing entries.
    '''
    # Mat73 doesn't support silently ignoring, so we have to actually filter
    # them out. Fortunately, we only care about top-level variables, so this
    # is easy to do.
    if only_include is not None:
        with h5py.File(filename) as f:
            only_include = [k for k in only_include if k in f]
    return mat73.loadmat(filename, only_include=only_include)


def load_raw(source, filename, only_include=None):
    'Load raw data from a .mat file under a data directory.'

    # We have to do this manually since we're using open() instead of
    # loadmat() directly.
    if not filename.endswith('.mat'):
        filename = filename + '.mat'

    full_path = data_dir(source) + '/' + filename

    try:
        with open(full_path, 'rb') as f:
            return scipy.io.loadmat(BytesIO(f.read()),
                                    variable_names=only_include)

    # This is horrific, but apparently none of the libraries for opening the
    # new Matlab format accept file-like objects. Since they require
    # a string path, if the file is nonlocal, I have to download it to
    # a tempfile first.
    except NotImplementedError:
        if full_path.startswith('s3://'):
            with tempfile.NamedTemporaryFile(suffix='.mat') as f:
                awswrangler.s3.download(full_path, f)
                return _load73(f.name, only_include)
        else:
            return _load73(full_path, only_include)


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
def get_raster(source, experiment, bin_size_ms, surrogate=None):
    if surrogate is None:
        return Raster(source, experiment, bin_size_ms)
    return get_raster(source, experiment, bin_size_ms).get_surrogate(surrogate)


class Raster:
    surrogates: dict[str, type] = {}
    surrogate_name = 'Real Data'

    def __init__(self, source, experiment, bin_size_ms):
        self.bin_size_ms = bin_size_ms
        self.source = source
        self.experiment = experiment
        self.tag = f'{source}_{experiment}_{bin_size_ms}ms'
        experiment, sl = exp_name_parts(experiment)

        if sl:
            raise NotImplementedError('Slicing not implemented yet.')

        # First try loading the data from .mat files, in either Mattia's or
        # Tal's format...
        try:
            # Save memory by only loading variables we'll actually use.
            mat = load_raw(source, experiment, only_include=[
                'spike_matrix', 'SUA', 'spike_train', 'fs', 'units',
                'spike_times', '#refs#'])

            # Mattia's data is formatted with something called a SUA, with
            # fixed sample rate of 1 kHz, or possibly the spike_matrix is
            # dumped directory in the root of the mat file.
            if 'spike_matrix' in mat or 'SUA' in mat:
                sm = (mat['SUA'][0,0]['spike_matrix'] if 'SUA' in mat 
                      else mat['spike_matrix'])
                self.raster, self._poprate, self.units = \
                    _raster_poprate_units_from_sm(sm.shape[1], bin_size_ms, sm)
                self.length_sec = sm.shape[1] / 1000
                self._burst_default_rms = 6.0

            # Tal and TJ's data is organized as units instead. For the UCSB
            # recordings, fs is stored and duration is not, so just assume
            # the recording was a whole number of seconds long by rounding
            # up the last spike time. For the ETH recordings, there is
            # a spike matrix to get the duration from, but use the raw spike
            # times because they're a bit higher resolution and the spike
            # matrix seems to have dropped some spikes.
            else:
                if 'spike_times' in mat:
                    self.units = []
                    for times in mat['spike_times']:
                        while len(times) == 1:
                            times = times[0]
                        self.units.append(times * 1e3)
                else:
                    self.units = [
                        (unit[0][0]['spike_train']/mat['fs']*1e3)[0,:]
                        for unit in mat['units'][0]]
                self.length_sec = np.ceil(max(
                    unit.max() for unit in self.units)/1e3)
                self.raster, self._poprate = _raster_poprate_from_units(
                    1e3*self.length_sec, self.bin_size_ms, self.units)
                self._burst_default_rms = 3.0

        # If those .mat files don't exist, instead load from Sury's phy
        # zips. This can't work if the data is in a mat format, though.
        except (OSError, FileNotFoundError):
            try:
                sd = read_phy_files(f'{data_dir(source)}/{experiment}.zip')
            except AssertionError as e:
                raise FileNotFoundError(
                    f'Failed to load {source} {experiment}: {e}') from None
            self.units = sd.train
            self.length_sec = sd.length / 1000
            self.raster = sd.raster(bin_size_ms).T
            self._poprate = sd.binned(1)
            self._burst_default_rms = None

        self.n_units = len(self.units)

    def spikes_within(self, start_ms, end_ms):
        'Return unit indices and spike times within a time window.'
        unitsub = [t[(t >= start_ms) & (t < end_ms)]
                   for t in self.units]
        times = np.hstack(unitsub)
        idces = np.hstack([[i]*len(t)
                           for i,t in enumerate(unitsub)
                           if len(t) > 0])
        return idces, times

    def coarse_rate(self):
        return self.poprate(20, 100)

    def fine_rate(self):
        return self.poprate(5, 5)

    def poprate(self, square_width_ms=0, gaussian_width_ms=None):
        '''
        Calculate population rate with a two-stage filter.

        The first stage is square and the second Gaussian. If one argument
        is provided, the same width is used for both filters. If either
        filter width is set to zero, that stage is skipped entirely.

        The width parameter of the Gaussian filter is five times its
        standard deviation because TJ uses an FIR Gaussian filter whose
        parameter is its support. The one here is actually IIR, but the
        difference is very small.
        '''
        ret = self._poprate * 1.0

        if square_width_ms > 0:
            ret = ndimage.uniform_filter1d(ret, square_width_ms)

        if gaussian_width_ms is None:
            gaussian_width_ms = square_width_ms
        if gaussian_width_ms > 0:
            ret = ndimage.gaussian_filter1d(ret, gaussian_width_ms/5)

        return ret

    def get_surrogate(self, which):
        return Raster.surrogates[which](self)

    def average_burst_bounds_ms(self, rms=None):
        "Average peak-relative start and end across all bursts."
        _, edges = self.find_burst_edges(rms=rms)
        return edges.mean(0)

    def find_burst_edges(self, rms=None):
        "Find the edges of the bursts in units of ms."
        # Find the peaks of the coarse rate.
        r_coarse = self.coarse_rate()
        if rms is None:
            rms = self._burst_default_rms
            if self._burst_default_rms is None:
                raise ValueError('No default rms value set for this data.')
        height = rms * np.sqrt(np.mean(r_coarse**2))
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

    def find_bursts(self, margins=None, rms=None):
        '''
        Find the locations of burst peaks in units of bins, filtered to only
        those with the desired margins.
        '''
        r_fine = self.fine_rate()
        peaks, edges = self.find_burst_edges(rms=rms)
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

    def observed_state_probs(self, h, burst_margins, burst_rms=None,
                             n_states=None):
        '''
        Return a probability distribution of the states in h over time
        relative to each of this Raster's burst peaks. Automatically
        determines how many states to include, but can be overridden by
        passing n_states.
        '''
        if n_states is None:
            n_states = h.max()+1
        peaks = self.find_bursts(margins=burst_margins, rms=burst_rms)
        lmargin, rmargin = burst_margins
        state_prob = np.zeros((n_states, rmargin-lmargin+1))

        for peak in peaks:
            peak_bin = int(round(peak))
            state_seq = h[peak_bin+lmargin:peak_bin+rmargin+1]
            for i, s in enumerate(state_seq):
                state_prob[s,i] += 1 / len(peaks)

        return state_prob

    def state_order(self, h, burst_margins, burst_rms=None, n_states=None):
        '''
        Return an order of the states based on the median burst-relative
        time at which they occur.

        Automatically determines how many states to include, but can be
        overridden by passing n_states.
        '''
        if n_states is None:
            n_states = h.max()+1
        peaks = self.find_bursts(margins=burst_margins, rms=burst_rms)
        lmargin, rmargin = burst_margins

        burst_relative_state_times = [[] for _ in range(n_states)]
        for peak in peaks:
            peak_bin = int(round(peak))
            state_seq = h[peak_bin+lmargin:peak_bin+rmargin+1]
            for i, s in enumerate(state_seq):
                burst_relative_state_times[s].append(i+lmargin)
        
        return np.argsort([np.median(times)
                           for times in burst_relative_state_times])


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
    dest._burst_default_rms = src._burst_default_rms
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
        rng = np.random.RandomState(2953)
        _steal_metadata(self, raster)

        self.units = [rng.rand(len(t))*self.length_sec*1e3
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
        rng = np.random.RandomState(2953)
        _steal_metadata(self, raster)

        counts = np.cumsum([0] + [len(unit) for unit in raster.units])
        times = np.hstack(raster.units)
        rng.shuffle(times)
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


@surrogate('rsm')
class RandSpikeMatrix(Raster):
    surrogate_name = 'Randomized Spike Matrix'
    def __init__(self, raster):
        '''
        Return a raster which preserves the population rate and mean firing
        rate of each neuron by randomly reallocating all spike times to
        different neurons, using resampling to maintain the invariant that
        no neuron spikes twice in the same millisecond.
        '''
        rng = np.random.RandomState(2953)
        _steal_metadata(self, raster)
        sm = _spike_matrix_from_units(
            1e3*self.length_sec, raster.units)

        rsm = np.zeros(sm.shape, int)
        weights = sm.sum(0)

        # Iterate over the units in order of how many spikes they have.
        n_spikeses = sm.sum(1)
        unit_order = np.argsort(n_spikeses)[::-1]
        unit_order = unit_order[n_spikeses[unit_order] > 0]

        # Choose spike times from the big list for each unit.
        for unit in unit_order:
            n_spikes = n_spikeses[unit]
            p = weights / weights.sum()
            rand_frames = rng.choice(
                rsm.shape[1], n_spikes, replace=False, p=p)
            weights[rand_frames] -= 1
            rsm[unit,rand_frames] = 1

        self.raster, self._poprate, self.units = \
            _raster_poprate_units_from_sm(1e3*self.length_sec,
                                          self.bin_size_ms, rsm)
