# Fit HMMs
#
# A script which takes a data source plus lists of experiment IDs, bin
# sizes, and numbers of hidden states, checks which of those models are not
# cached, and either adds them to the HMM job queue or just runs them
# locally, depending whether the option --local is provided.
import argparse
import fnmatch
import itertools
import os
import sys

import numpy as np
from braingeneers.iot.messaging import MessageBroker
from tqdm import tqdm

import hmmsupport


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
        prog="fit_hmms", description="Fit HMMs locally or on NRP."
    )
    parser.add_argument("source")
    parser.add_argument("exp", type=lambda x: x if "*" in x else ensure_list(x))
    parser.add_argument("bin_sizes", type=parse_range_str)
    parser.add_argument("n_stateses", type=parse_range_str)
    parser.add_argument("surrs", default=["real"], nargs="?", type=ensure_list)
    parser.add_argument(
        "-n", "--dryrun", action="store_true", help="just print what would be done"
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        help="fit HMMs locally instead of queueing jobs",
    )
    parser.add_argument(
        "--clear-queue",
        action="store_true",
        help="clear the queue before adding new jobs",
    )
    parser.add_argument("--pbar", action="store_true", help="show a progress bar")
    args = parser.parse_args()

    if not args.local and not os.environ.get("S3_USER"):
        print("$S3_USER must be defined.", file=sys.stderr)
        sys.exit(1)

    # Support *. Can't be part of the type because it depends on source.
    if "*" in args.exp:
        args.exp = fnmatch.filter(hmmsupport.all_experiments(args.source), args.exp)

    # Verbosely print the full parameter set.
    print(f"Fitting HMMs for experiments:")
    for exp, surr in itertools.product(args.exp, args.surrs):
        print(f"  {args.source}/{exp}[{surr}]")
    print("Will use K in", args.n_stateses)
    print("Will use T in", args.bin_sizes)

    # Check which parameters actually need re-run.
    print("Must fit...")
    needs_run = []
    needs_check = list(
        itertools.product(args.exp, args.bin_sizes, args.n_stateses, args.surrs)
    )
    for p in tqdm(needs_check, total=len(needs_check), disable=not args.pbar):
        if not hmmsupport.is_cached(args.source, *p):
            needs_run.append(p)
            tqdm.write(f"  {args.source}/{p[0]}[{p[3]}] with T={p[1]}ms, K={p[2]}.")

    if not needs_run:
        print("Nothing. All HMMs are already cached.")
        sys.exit()

    print(f"({len(needs_run)} HMMs in total.)")

    # If this is a dry run, don't actually bother queueing anything.
    if args.dryrun:
        sys.exit()

    # If local, don't use MQTT, just do all the fits in a loop here.
    elif args.local:
        for exp, bin_size_ms, n_states, surrogate in needs_run:
            hmmsupport.get_fitted_hmm(
                args.source,
                exp,
                bin_size_ms,
                n_states,
                surrogate,
                verbose=True,
            )
        sys.exit()

    # Get the MQTT queue, clearing it if requested.
    queue_name = f"{os.environ.get('S3_USER')}/hmm-job-queue"
    mb = MessageBroker()
    if args.clear_queue:
        mb.delete_queue(queue_name)
    q = mb.get_queue(queue_name)

    # Add all the jobs to the queue.
    for exp, bin_size_ms, n_states, surrogate in needs_run:
        q.put(
            dict(
                retries_allowed=3,
                params=dict(
                    source=args.source,
                    exp=exp,
                    bin_size_ms=bin_size_ms,
                    n_states=n_states,
                    surrogate=surrogate,
                ),
            )
        )
