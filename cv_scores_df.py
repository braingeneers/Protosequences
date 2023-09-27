import joblib
import numpy as np
import pandas as pd

from hmmsupport import all_experiments, cv_scores, memoize


@memoize
def cv_scores_df():
    source = "org_and_slice"
    # Note that these *have* to be np.int64 because joblib uses argument hashes that
    # are different for different integer types!
    params = [
        (exp, np.int64(bin_size_ms), np.int64(n_states))
        for exp in all_experiments(source)
        if exp.startswith("L")
        for bin_size_ms in [10, 20, 30, 50, 70, 100]
        for n_states in range(10, 51)
    ]

    def cache_params(i, p):
        if cv_scores.check_call_in_cache(source, *p):
            return cv_scores(source, *p)
        else:
            print(f"{i}/{len(params)} {p} MISSING")

    scores = joblib.Parallel(backend="threading", n_jobs=10, verbose=10)(
        joblib.delayed(cache_params)(i, p) for i, p in enumerate(params)
    )

    cv_scoreses = dict(zip(params, scores))

    df = []
    for (exp, bin_size_ms, num_states), scores in cv_scoreses.items():
        organoid = exp.split("_", 1)[0]

        # Combine those into dataframe rows, one per score rather than one per file
        # like a db normalization because plotting will expect that later anyway.
        df.extend(
            dict(
                organoid=organoid,
                bin_size=bin_size_ms,
                states=num_states,
                ll=ll,
                surr_ll=surr_ll,
                train_ll=train_ll,
                delta_ll=ll - surr_ll,
            )
            for ll, surr_ll, train_ll in zip(
                scores["validation"], scores["surrogate"], scores["training"]
            )
        )
    return pd.DataFrame(sorted(df, key=lambda row: int(row["organoid"][1:])))


# This freaky boilerplate is necessary because joblib cache paths include the package
# name. It catches the case where the function is cached by running this file as a
# script: in that case, we manually import the packaged version of the function, and it
# gets cached under the packaged name.
if __name__ == "__main__":
    from cv_scores_df import df
else:
    df = cv_scores_df()
