from hmmsupport import become_worker, get_fitted_hmm


def fit_hmm(job):
    "Fit an HMM with the job's parameters, retrying if it fails."
    try:
        get_fitted_hmm(**job.params)

    except ZeroDivisionError:
        if job.requeue():
            print("Optimization failed, retrying.")
        else:
            print("Optimization failed.")


become_worker("hmm", fit_hmm)
