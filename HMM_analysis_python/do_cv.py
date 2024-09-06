# Do Cross Validation
#
# A script which takes a data source plus lists of experiment IDs, bin
# sizes, and numbers of hidden states, checks which ones are already on
# S3, and adds the rest to the job queue for cross-validation.
import argparse
import fnmatch
import os
import sys

import numpy as np
from braingeneers.iot.messaging import MessageBroker
from joblib import Parallel, delayed
from tqdm import tqdm

from hmmsupport import all_experiments, cv_scores


def ensure_list(str_str):
    if "," in str_str:
        return str_str.split(",")
    else:
        return [str_str]


def parse_range_str(range_str):
    if "," in range_str:
        return np.array([int(x) for x in range_str.split(",")])
    elif ":" in range_str:
        parts = range_str.split(":", 2)
        step = int(parts[2]) if len(parts) == 3 else None
        return np.arange(int(parts[0]), int(parts[1]) + 1, step)
    elif "-" in range_str:
        start, end = range_str.split("-")
        return np.arange(int(start), int(end) + 1)
    else:
        return np.array([int(range_str)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="fit_hmms", description="Queue cross-validation for HMMs."
    )
    parser.add_argument("source")
    parser.add_argument("exp", type=ensure_list)
    parser.add_argument("bin_sizes", type=parse_range_str)
    parser.add_argument("n_stateses", type=parse_range_str)
    parser.add_argument(
        "-n", "--dryrun", action="store_true", help="just print what would be done"
    )
    parser.add_argument(
        "--clear-queue",
        action="store_true",
        help="clear the queue before adding new jobs",
    )
    args = parser.parse_args()

    if not os.environ.get("S3_USER"):
        print("$S3_USER must be defined.", file=sys.stderr)
        sys.exit(1)

    # Turn a list whose entries may contain wildcards into a simple list.
    all_exps = all_experiments(args.source)
    exps = []
    for exp in args.exp:
        if "*" in exp:
            exps.extend(fnmatch.filter(all_exps, exp))
        else:
            exps.append(exp)

    # Verbosely print the full parameter set.
    print("Cross-validating HMMs on the following experiments:")
    for exp in exps:
        print(f"  {args.source}/{exp}")
    print("Will use K in", args.n_stateses)
    print("Will use T in", args.bin_sizes)

    # Check which parameters actually need re-run.
    print("Must fit...")
    all_params = [
        (exp, bin_size_ms, n_states)
        for exp in exps
        for bin_size_ms in args.bin_sizes
        for n_states in args.n_stateses
    ]

    with tqdm(total=len(all_params)) as pbar:

        def is_missing(p):
            missing = not cv_scores.check_call_in_cache(args.source, *p)
            pbar.update()
            if missing:
                tqdm.write(f"  {args.source}/{p[0]} with T={p[1]}ms, K={p[2]}.")
            return missing

        needs_run = Parallel(n_jobs=-1, backend="threading")(
            delayed(is_missing)(p) for p in all_params
        )

    job_params = [p for p, missing in zip(all_params, needs_run) if missing]

    if not job_params:
        print("Nothing. All CV jobs are already done.")
        sys.exit()

    # If this is a dry run, don't actually bother queueing anything.
    if args.dryrun:
        print(f"Would have fit {len(job_params)} CV jobs.")
        sys.exit()

    print(f"Queueing {len(job_params)} CV jobs...")

    # Get the MQTT queue, clearing it if requested.
    queue_name = f"{os.environ.get('S3_USER')}/cv-job-queue"
    mb = MessageBroker()
    if args.clear_queue:
        mb.delete_queue(queue_name)
    q = mb.get_queue(queue_name)

    # Add all the jobs to the queue.
    for exp, bin_size_ms, n_states in job_params:
        q.put(
            dict(
                retries_allowed=3,
                params=dict(
                    source=args.source,
                    exp=exp,
                    bin_size_ms=bin_size_ms,
                    n_states=n_states,
                ),
            )
        )
