from worker import become_worker
from hmmsupport import get_fitted_hmm

def fit_hmm(job):
    "Fit an HMM with the job's parameters, retrying if it fails."
    try:
        get_fitted_hmm(**job.params, verbose=True)

    except ZeroDivisionError:
        s = job.params["source"]
        e = job.params["exp"]
        r = job.params["surrogate"]
        T = job.params["bin_size_ms"]
        K = job.params["n_states"]
        if job.requeue():
            print(f"Retrying {s}/{e}[{r}] with {T=}ms, {K=}.")
        else:
            print(f"Failed {s}/{e}[{r}] with {T=}ms, {K=}!")

become_worker("hmm", fit_hmm)
