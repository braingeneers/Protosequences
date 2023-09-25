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
    cv_scoreses = {}
    total = len(params)
    for i,p in enumerate(params):
        if cv_scores.check_call_in_cache(source, *p):
            cv_scoreses[p] = cv_scores(source, *p)
            status = "OK"
        else:
            status = "NOT IN CACHE!"
        print(f"{i}/{total} {p} {status}")

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
                score=value,
            )
            for value in scores["validation"] - scores["surrogate"]
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
