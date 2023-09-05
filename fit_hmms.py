# Fit HMMs
#
# A script which takes a data source plus lists of experiment IDs, bin
# sizes, and numbers of hidden states, checks which of those models are not
# cached, and either adds them to the HMM job queue or just runs them
# locally, depending whether the option --local is provided.
import os
import sys
import itertools
import numpy as np
import hmmsupport
import argparse
from braingeneers.iot.messaging import MessageBroker


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


def hmm_method_type(name):
    if name not in hmmsupport._HMM_METHODS:
        raise ValueError
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog="fit_hmms",
            description="Fit HMMs locally or on NRP.")
    parser.add_argument('source')
    parser.add_argument('exp',
            type=lambda x: x if x == "*" else ensure_list(x))
    parser.add_argument('bin_sizes', type=parse_range_str)
    parser.add_argument('n_stateses', type=parse_range_str)
    parser.add_argument('surrs', default=["real"], type=ensure_list)
    parser.add_argument('library', default="default", nargs='?',
            type=hmm_method_type)
    parser.add_argument('-n', '--dryrun', action='store_true')
    parser.add_argument('-l', '--local', action='store_true')
    parser.add_argument('--clear-queue', action='store_true')
    args = parser.parse_args()

    if not args.local and not os.environ.get("S3_USER"):
        print("$S3_USER must be defined.", file=sys.stderr)
        sys.exit(1)

    # Can't be part of the type because it depends on source.
    if args.exp == "*":
        args.exp = hmmsupport.all_experiments(args.source)

    # Verbosely print the full parameter set.
    print(f"Fitting HMMs using {args.library} for experiments:")
    for exp, surr in itertools.product(args.exp, args.surrs):
        print(f"  {args.source}/{exp}[{surr}]")
    print("Will use K in", args.n_stateses)
    print("Will use T in", args.bin_sizes)

    # Check which parameters actually need re-run.
    needs_run = [
        p
        for p in itertools.product(args.exp, args.bin_sizes,
            args.n_stateses, args.surrs)
        if not hmmsupport.is_cached(args.source, *p, library=args.library)
    ]
    if not needs_run:
        print("All HMMs are already cached.")
        sys.exit(0)

    print(f"{len(needs_run)} HMMs need to be fit:")
    for exp, T, K, surrogate in needs_run:
        print(f"  {args.source}/{exp}[{surrogate}] with {T=}ms, {K=}.")

    # If this is a dry run, don't actually bother queueing anything.
    if args.dryrun:
        sys.exit()

    # If local, don't use MQTT, just do all the fits in a loop here.
    elif args.local:
        for exp, bin_size_ms, n_states, surrogate in needs_run:
            hmmsupport.get_fitted_hmm(
                args.source, exp, bin_size_ms, n_states, surrogate,
                args.library, verbose=True
            )
        sys.exit(0)

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
                    library=args.library,
                ),
            )
        )
