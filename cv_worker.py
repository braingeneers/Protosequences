import os
import pickle
import numpy as np
from ssm import HMM
from braingeneers.utils.smart_open_braingeneers import open
from hmmsupport import get_raster, FIT_ATOL, FIT_N_ITER
from worker import become_worker

CV_N_FOLDS = 5
BASEPATH = f"s3://braingeneers/personal/{os.environ['S3_USER']}/cache/cv_scores"


def s3_filename(source, exp, bin_size_ms, n_states):
    return f"{BASEPATH}/{source}_{exp}_{bin_size_ms}ms_K{n_states}_real.pickle"


def cross_validate(job):
    """
    Run cross-validation for a given parameter set, and store the results to a
    pickled dict on S3 with keys `training`, `validation`, and `surrogate`,
    each a 5-element array representing the performance of a single trained
    model on the training and validation sets, as well as on the validation
    portion of the randomized surrogate version of the training set.

    Adapted from `ssm.model_selection.cross_val_scores()`.
    """
    source = job.params["source"]
    exp = job.params["exp"]
    bin_size_ms = job.params["bin_size_ms"]
    n_states = job.params["n_states"]
    raster = get_raster(source, exp, bin_size_ms)
    hmm = HMM(K=n_states, D=raster._raster.shape[1], observations="poisson")

    # Allocate space for train and test log-likelihoods, as well as LL on the surrogate
    # data of the same shape as the validation set.
    ret = dict(training=np.empty(CV_N_FOLDS),
               validation=np.empty(CV_N_FOLDS),
               surrogate=np.empty(CV_N_FOLDS))

    data = raster._raster
    fake_data = raster.randomized()._raster

    for r in range(CV_N_FOLDS):
        # Create mask for training data.
        train_mask = np.ones_like(data, dtype=bool)

        # Determine number of heldout points.
        n_total = np.sum(train_mask)
        obs_inds = np.argwhere(train_mask)
        heldout_num = int(n_total * 0.1)

        # Randomly hold out speckled data pattern.
        heldout_flat_inds = np.random.choice(n_total, heldout_num, replace=False)

        # Create training mask.
        i, j = obs_inds[heldout_flat_inds].T
        train_mask[i, j] = False

        # Fit model with training mask.
        try:
            hmm.fit(
                data,
                masks=train_mask,
                tolerance=FIT_ATOL,
                num_iters=FIT_N_ITER,
                verbose=2,
            )

        except ZeroDivisionError:
            if job.requeue():
                print("Optimization failed, retrying.")
            else:
                print("Optimization failed.")
            return

        # Compute log-likelihood on full, training, and surrogate training data.
        full_ll = hmm.log_likelihood(data)
        train_ll = hmm.log_likelihood(data, masks=train_mask)
        surr_ll = hmm.log_likelihood(fake_data, masks=~train_mask)

        # Total number of training and observed datapoints.
        n_train = train_mask.sum()

        # Calculate normalized log-likelihood scores.
        ret["training"][r] = train_ll / n_train
        ret["validation"][r] = (full_ll - train_ll) / (n_total - n_train)
        ret["surrogate"][r] = surr_ll / (n_total - n_train)

    with open(s3_filename(source, exp, bin_size_ms, n_states), "wb") as f:
        pickle.dump(ret, f)


if __name__ == "__main__":
    become_worker("cv", cross_validate)
