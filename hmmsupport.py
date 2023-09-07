import functools
import glob
import os
import pickle
import re
import sys
from contextlib import contextmanager
from io import BytesIO

import awswrangler
import mat73
import numpy as np
import scipy.io
from braingeneers.analysis import load_spike_data, SpikeData
from braingeneers.utils.smart_open_braingeneers import open
from scipy import signal, ndimage


def s3_isdir(path):
    try:
        next(awswrangler.s3.list_objects(path, chunked=True))
        return True
    except StopIteration:
        return False


DATA_DIR = "data"


@functools.lru_cache
def data_dir(source):
    path = os.path.join(DATA_DIR, source)
    if os.path.isdir(path):
        return path
    elif path.startswith("s3://") and s3_isdir(path):
        return path

    personal_s3 = "s3://braingeneers/personal/atspaeth/data/" + source
    if s3_isdir(personal_s3):
        return personal_s3

    main_s3 = f"s3://braingeneers/ephys/{source}/derived/kilosort2"
    if s3_isdir(main_s3):
        return main_s3

    raise FileNotFoundError(f"Could not find data for {source}")


def all_experiments(source):
    path = data_dir(source)
    if path.startswith("s3://"):
        paths = awswrangler.s3.list_objects(path + "/")
    else:
        paths = glob.glob(os.path.join(path, "*"))
    return sorted({os.path.splitext(os.path.basename(x))[0] for x in paths})


def multi_source_experiments(*sources):
    ret = []
    for source in sources:
        for experiment in all_experiments(source):
            ret.append((source, experiment))
    return ret


CACHE_DIR = ".cache"
S3_USER = os.environ.get("S3_USER")
S3_CACHE = f"s3://braingeneers/personal/{S3_USER}/cache"
CACHE_ROOTS = [
    root_path
    for (root_path, valid) in [
        (CACHE_DIR, os.path.isdir(CACHE_DIR)),
        (S3_CACHE, S3_USER is not None),
    ]
    if valid
]


def _store(obj, path):
    """
    Pickle an object to a path.

    If the path is local, ensure that it exists before trying to open the
    file.
    """
    iss3 = path.startswith("s3://")
    try:
        if not iss3:
            dirname = os.path.dirname(path)
            os.makedirs(dirname, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        verb = "upload" if iss3 else "save"
        print(f"Failed to {verb} {path}: {e}", file=sys.stderr)


class Cache:
    def __init__(self, func):
        self.wrapped = func
        functools.update_wrapper(self, func)

    def _cache_names(self, source, exp, bin_size_ms, n_states, surrogate):
        "Local cache file and S3 object cache path."
        key = f"{source}_{exp}_{bin_size_ms}ms_{n_states}S_{surrogate}.pkl"
        file = os.path.join(CACHE_DIR, self.__name__, key)
        s3_object = "/".join((S3_CACHE, self.__name__, key))
        return file, s3_object

    def _cache_path(self, *args):
        "The nearest possible cache path to read from, None if uncached."
        filename, s3_object = self._cache_names(*args)

        if os.path.isdir(CACHE_DIR) and os.path.isfile(filename):
            return filename
        elif S3_USER and awswrangler.s3.does_object_exist(s3_object):
            return s3_object
        else:
            return None

    def is_cached(self, *args):
        "Whether the given parameters are cached."
        return self._cache_path(*args) is not None

    def get_cached(self, *args):
        """
        Get the cached object, or None if unavailable.

        Additionally, if restoring the object from S3 when a local cache is
        present, add the object to the local cache.
        """
        path = self._cache_path(*args)

        if path is None:
            return None

        with open(path, "rb") as f:
            ret = pickle.load(f)

        if path.startswith("s3://") and os.path.isdir(CACHE_DIR):
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
        if path.startswith("/"):
            figdir.dir = path
        else:
            figdir.dir = os.path.join("figures", path)
        if not os.path.exists(figdir.dir):
            os.makedirs(figdir.dir)
    return os.path.abspath(figdir.dir)


figdir("")


FIT_ATOL = 1e-3
FIT_N_ITER = 5000

_HMM_METHODS = {}


class HMMMethod:
    def __init__(self, library, fit, states, cross_validate):
        self.library = library
        self.fit = fit
        self.states = states
        self.cross_validate = cross_validate
        _HMM_METHODS[library] = self


@Cache
def _ssm_hmm(
    source,
    exp,
    bin_size_ms,
    n_states,
    surrogate,
    verbose=False,
    atol=FIT_ATOL,
    n_iter=FIT_N_ITER,
):
    "Fit an HMM to data with SSM and return the model."
    from ssm import HMM as SSMHMM

    r = get_raster(source, exp, bin_size_ms, surrogate)
    hmm = SSMHMM(K=n_states, D=r._raster.shape[1], observations="poisson")
    hmm.fit(r._raster, verbose=2 if verbose else 0, tolerance=atol, num_iters=n_iter)
    return hmm


def _ssm_states(hmm, raster):
    "Return the most likely state sequence for the given raster."
    return hmm.most_likely_states(raster)


def _ssm_cv(hmm, raster, n_folds, atol=FIT_ATOL, n_iter=FIT_N_ITER, verbose=False):
    "Return validation and training scores on cross-validation folds."
    from ssm.model_selection import cross_val_scores

    return cross_val_scores(
        hmm,
        raster,
        n_repeats=n_folds,
        tolerance=atol,
        num_iters=n_iter,
        verbose=2 if verbose else 0,
    )


SSMFit = HMMMethod("ssm", _ssm_hmm, _ssm_states, _ssm_cv)


default_method = os.environ.get("HMM_METHOD", "ssm")
_HMM_METHODS["default"] = _HMM_METHODS[default_method]


def is_cached(source, exp, bin_size_ms, n_states, surrogate="real", library="default"):
    "Return whether the given model is cached."
    return _HMM_METHODS[library].fit.is_cached(
        source, exp, bin_size_ms, n_states, surrogate
    )


def get_fitted_hmm(
    source,
    exp,
    bin_size_ms,
    n_states,
    surrogate="real",
    recompute_ok=True,
    library="default",
    verbose=False,
):
    if verbose:
        print(f"Running {source}/{exp}:{bin_size_ms}ms, K={n_states}")
    params = source, exp, bin_size_ms, n_states, surrogate
    method = _HMM_METHODS[library]
    if recompute_ok:
        return method.fit(*params, verbose=verbose)
    else:
        return method.fit.get_cached(*params)


class Model:
    def __init__(
        self,
        source,
        exp,
        bin_size_ms,
        n_states,
        surrogate="real",
        library="default",
        verbose=False,
        recompute_ok=False,
    ):
        # Retrieve the (hopefully cached) model.
        self._hmm = get_fitted_hmm(
            source,
            exp,
            bin_size_ms,
            n_states,
            surrogate,
            library=library,
            verbose=verbose,
            recompute_ok=recompute_ok,
        )

        # Save metadata.
        self.source = source
        self.exp = exp
        self.bin_size_ms = bin_size_ms
        self.n_states = n_states
        self.surrogate = surrogate
        self.library = library

    def compute_consistency(self, raster, metrics):
        """
        Compute an n_states x n_units array indicating how likely a unit is
        to have nonzero firings in each time bin of a given state.
        """
        abs_margin = int(1e3 / self.bin_size_ms)
        self.burst_margins = -abs_margin, abs_margin

        self.h = self.states(raster)
        scores = np.array(
            [(raster._raster[self.h == i, :] > 0).mean(0) for i in range(self.n_states)]
        )
        self.unit_order = np.int32(metrics["mean_rate_ordering"].flatten()) - 1
        self.state_order = raster.state_order(
            self.h, self.burst_margins, n_states=self.n_states
        )
        scores[np.isnan(scores)] = 0
        self.consistency = scores[:, self.unit_order][self.state_order, :]

    def states(self, raster):
        return _HMM_METHODS[self.library].states(self._hmm, raster._raster)

    def cross_validate(self, raster, n_folds):
        return _HMM_METHODS[self.library].cross_validate(
            self._hmm, raster._raster, n_folds
        )

    def dump(self, path):
        """
        Dump the parameters of this HMM to a .mat file.
        """
        mat = dict(
            state_means=np.exp(self._hmm.observations.log_lambdas)[
                self._hmm.state_order, :
            ],
            state_sequence=[
                np.nonzero(self._hmm.state_order == s)[0][0] for s in self.states()
            ],
            n_states=self.hmm.K,
            bin_size_ms=self.hmm.bin_size_ms,
        )
        with open(os.path.join(figdir(), path), "wb") as f:
            scipy.io.savemat(f, mat)


@contextmanager
def figure(name, save_args={}, save=True, save_exts=["png"], **kwargs):
    "Create a named figure and save it when done."
    import matplotlib.pyplot as plt

    f = plt.figure(name, **kwargs)
    try:
        f.clf()
    except Exception:
        plt.close(f)
        f = plt.figure(name, **kwargs)

    yield f

    if save:
        fname = name.lower().strip().replace(" ", "-")
        for ext in save_exts:
            if ext[0] != ".":
                ext = "." + ext
            path = os.path.join(figdir(), fname + ext)
            f.savefig(path, **save_args)


def load_raw(source, filename, only_include=None, error=True):
    """
    Load raw data from a .mat file under a data directory. If the file cannot be
    found, return None if error=False, otherwise raise an exception.
    Optionally only include a subset of variables, but for now this still loads
    the entire file into memory.
    """

    # We have to do this manually since we're using open() instead of
    # loadmat() directly.
    if not filename.endswith(".mat"):
        filename = filename + ".mat"

    # Load into memory because these methods use a ton of random
    # access and it gets really slow with smart_open.
    try:
        with open(data_dir(source) + "/" + filename, "rb") as f:
            contents = BytesIO(f.read())
        return scipy.io.loadmat(contents, variable_names=only_include)
    except NotImplementedError:
        return mat73.loadmat(contents, only_include=only_include, verbose=False)
    except (OSError, FileNotFoundError):
        if error:
            raise


def load_metrics(exp, only_include=None, error=True):
    "Use load_raw to get the metrics for an experiment."
    filename = exp.split("_", 1)[0] + "_single_recording_metrics.mat"
    return load_raw("metrics", filename, only_include, error)


@functools.lru_cache
def get_raster(source, experiment, bin_size_ms, surrogate=None):
    if surrogate == "rsm":
        return get_raster(source, experiment, bin_size_ms).randomized(seed=2953)
    return Raster(source, experiment, bin_size_ms)


class Raster(SpikeData):
    def __init__(self, source, experiment, bin_size_ms):
        # First try loading the data from .mat files, in either Mattia's or
        # Tal's format...
        try:
            # Save memory by only loading variables we'll actually use.
            mat = load_raw(
                source,
                experiment,
                only_include=[
                    "spike_matrix",
                    "t_spk_mat",
                    "SUA",
                    "spike_train",
                    "fs",
                    "units",
                    "spike_times",
                    "#refs#",
                ],
            )

            # Mattia's data is formatted with something called a SUA, with
            # fixed sample rate of 1 kHz, or possibly the spike_matrix is
            # dumped directory in the root of the mat file.
            if "spike_matrix" in mat or "SUA" in mat:
                sm = (
                    mat["SUA"][0, 0]["spike_matrix"]
                    if "SUA" in mat
                    else mat["spike_matrix"]
                )
                self._burst_default_rms = 6.0
                idces, times = np.nonzero(sm)
                units = [times[idces == i] for i in range(sm.shape[0])]
                length = sm.shape[1] * 1.0

            # Tal and TJ's data is organized as units instead. For the UCSB
            # recordings, fs is stored and duration is not, so just assume
            # the recording was a whole number of seconds long by rounding
            # up the last spike time. For the ETH recordings, there is
            # a spike matrix to get the duration from, but use the raw spike
            # times because they're a bit higher resolution and the spike
            # matrix seems to have dropped some spikes.
            else:
                if "t_spk_mat" in mat:
                    times, idces = np.nonzero(mat["t_spk_mat"])
                    units = [
                        times[idces == i] for i in range(mat["t_spk_mat"].shape[1])
                    ]
                elif "spike_times" in mat:
                    units = []
                    for times in mat["spike_times"]:
                        # Breaks L1_t_spk_mat_sorted
                        while len(times) == 1:
                            times = times[0]
                        units.append(times * 1e3)
                else:
                    units = [
                        (unit[0][0]["spike_train"] / mat["fs"] * 1e3)[0, :]
                        for unit in mat["units"][0]
                    ]
                length = 1e3 * np.ceil(max(unit.max() for unit in units) / 1e3)
                self._burst_default_rms = 3.0

        # If those .mat files don't exist, instead load from Sury's phy
        # zips. This can't work if the data is in a mat format, though.
        except OSError:
            try:
                full_path = f"{data_dir(source)}/{experiment}.zip"
                sd = load_spike_data(None, full_path=full_path)
            except OSError:
                raise FileNotFoundError(f"Experiment {source}/{experiment} not found")
            self._burst_default_rms = None
            units, length = sd.train, sd.length

        # Delegate out to the part that's useful to extract for subclass
        # constructors.
        self._init(source, experiment, bin_size_ms, units, length)

    def _init(self, source, experiment, bin_size_ms, units, length):
        """
        The boilerplate that should be used in subclass constructors to make
        sure the correct attributes get assigned.
        """
        self.bin_size_ms = bin_size_ms
        self.source = source
        self.experiment = experiment
        self.tag = f"{source}_{experiment}_{bin_size_ms}ms"
        super().__init__(units, length=length)
        self._poprate = self.binned(1)
        self._raster = self.raster(bin_size_ms).T

    def coarse_rate(self):
        return self.poprate(20, 100)

    def fine_rate(self):
        return self.poprate(5, 5)

    def poprate(self, square_width_ms=0, gaussian_width_ms=None):
        """
        Calculate population rate with a two-stage filter.

        The first stage is square and the second Gaussian. If one argument
        is provided, the same width is used for both filters. If either
        filter width is set to zero, that stage is skipped entirely.

        The width parameter of the Gaussian filter is five times its
        standard deviation because TJ uses an FIR Gaussian filter whose
        parameter is its support. The one here is actually IIR, but the
        difference is very small.
        """
        ret = self._poprate * 1.0

        if square_width_ms > 0:
            ret = ndimage.uniform_filter1d(ret, square_width_ms)

        if gaussian_width_ms is None:
            gaussian_width_ms = square_width_ms
        if gaussian_width_ms > 0:
            ret = ndimage.gaussian_filter1d(ret, gaussian_width_ms / 5)

        return ret

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
                raise ValueError("No default rms value set for this data.")
        height = rms * np.sqrt(np.mean(r_coarse**2))
        peaks_ms = signal.find_peaks(r_coarse, height=height, distance=700)[0]

        # Descend from those peaks in both directions to find the first
        # points where the coarse rate is below 10% of the peak value.
        n = len(peaks_ms)
        edges = np.zeros((n, 2), int)
        for i, peak in enumerate(peaks_ms):
            min_height = 0.1 * r_coarse[peak]
            while peak + edges[i, 0] >= 0 and r_coarse[peak + edges[i, 0]] > min_height:
                edges[i, 0] -= 1
            while (
                peak + edges[i, 1] < len(r_coarse)
                and r_coarse[peak + edges[i, 1]] > min_height
            ):
                edges[i, 1] += 1
        return peaks_ms, edges

    def find_bursts(self, margins=None, rms=None):
        """
        Find the locations of burst peaks in units of bins, filtered to only
        those with the desired margins.
        """
        r_fine = self.fine_rate()
        peaks, edges = self.find_burst_edges(rms=rms)
        peaks = (
            np.array(
                [
                    peaks[i]
                    + edges[i, 0]
                    + np.argmax(r_fine[peaks[i] + edges[i, 0] : peaks[i] + edges[i, 1]])
                    for i in range(len(peaks))
                ]
            )
            / self.bin_size_ms
        )

        # Filter out peaks too close to the edges of the recording.
        if margins is not None:
            try:
                lmargin, rmargin = margins
            except TypeError:
                lmargin, rmargin = -margins, margins
            peaks = peaks[
                (peaks + lmargin >= 0) & (peaks + rmargin < self._raster.shape[0])
            ]

        return peaks

    def observed_state_probs(self, h, burst_margins, burst_rms=None, n_states=None):
        """
        Return a probability distribution of the states in h over time
        relative to each of this Raster's burst peaks. Automatically
        determines how many states to include, but can be overridden by
        passing n_states.
        """
        if n_states is None:
            n_states = h.max() + 1
        peaks = self.find_bursts(margins=burst_margins, rms=burst_rms)
        lmargin, rmargin = burst_margins
        state_prob = np.zeros((n_states, rmargin - lmargin + 1))

        for peak in peaks:
            peak_bin = int(round(peak))
            state_seq = h[peak_bin + lmargin : peak_bin + rmargin + 1]
            for i, s in enumerate(state_seq):
                state_prob[s, i] += 1 / len(peaks)

        return state_prob

    def state_order(self, h, burst_margins, burst_rms=None, n_states=None):
        """
        Return an order of the states based on the median burst-relative
        time at which they occur.

        Automatically determines how many states to include, but can be
        overridden by passing n_states.
        """
        if n_states is None:
            n_states = h.max() + 1
        peaks = self.find_bursts(margins=burst_margins, rms=burst_rms)
        lmargin, rmargin = burst_margins

        burst_relative_state_times = [[] for _ in range(n_states)]
        for peak in peaks:
            peak_bin = int(round(peak))
            state_seq = h[peak_bin + lmargin : peak_bin + rmargin + 1]
            for i, s in enumerate(state_seq):
                burst_relative_state_times[s].append(i + lmargin)

        return np.argsort([np.median(times) for times in burst_relative_state_times])

    def randomized(self, bin_size=1.0, seed=2953):
        "As SpikeData.randomized(), but return a Raster."
        sd = super().randomized(bin_size, seed)
        ret = self.__class__.__new__(self.__class__)
        ret._init(self.source, self.experiment, self.bin_size_ms, sd.train, sd.length)
        ret._burst_default_rms = self._burst_default_rms
        return ret
