import os
import pickle
from ssm import HMM
from braingeneers.utils.smart_open_braingeneers import open
from hmmsupport import get_raster, _ssm_cv
from worker import become_worker

BASEPATH = f"s3://braingeneers/personal/{os.environ['S3_USER']}/cache/cv_scores"


def cross_validate(job):
    "Run cross-validation for a given parameter set."
    source = job.params["source"]
    exp = job.params["exp"]
    bin_size_ms = job.params["bin_size_ms"]
    n_states = job.params["n_states"]
    raster = get_raster(source, exp, bin_size_ms)
    hmm = HMM(K=n_states, D=raster._raster.shape[1], observations="poisson")
    try:
        training, validation, surrogate = _ssm_cv(hmm, raster, 5, verbose=True)
        tag = f"{source}_{exp}_{bin_size_ms}ms_K{n_states}_real"
        with open(f"{BASEPATH}/{tag}.pickle", "wb") as f:
            pickle.dump(
                dict(validation=validation, training=training, surrogate=surrogate), f
            )
    except ZeroDivisionError:
        if job.requeue():
            print("Optimization failed, retrying.")
        else:
            print("Optimization failed.")


become_worker("cv", cross_validate)
