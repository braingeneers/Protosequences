# Do Cross Validation
#
# A script which takes a data source plus lists of experiment IDs, bin
# sizes, and numbers of hidden states, and adds them all to the job queue for
# cross-validation.
import os
import sys
import itertools
import numpy as np
from hmmsupport import _HMM_METHODS, all_experiments
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
    if name not in _HMM_METHODS:
        raise ValueError
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="fit_hmms", description="Queue cross-validation for HMMs."
    )
    parser.add_argument("source")
    parser.add_argument("exp", type=lambda x: x if x == "*" else ensure_list(x))
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
    parser.add_argument("--pbar", action="store_true", help="show a progress bar")
    args = parser.parse_args()

    if not os.environ.get("S3_USER"):
        print("$S3_USER must be defined.", file=sys.stderr)
        sys.exit(1)

    # Can't be part of the type because it depends on source.
    if args.exp == "*":
        args.exp = all_experiments(args.source)

    # Verbosely print the full parameter set.
    print("Cross-validating HMMs on the following experiments:")
    for exp in args.exp:
        print(f"  {args.source}/{exp}")
    print("Will use K in", args.n_stateses)
    print("Will use T in", args.bin_sizes)

    # Check which parameters actually need re-run.
    needs_run = list(
        itertools.product(args.exp, args.bin_sizes, args.n_stateses)
    )

    print(f"Queueing {len(needs_run)} CV jobs...")

    # If this is a dry run, don't actually bother queueing anything.
    if args.dryrun:
        sys.exit()

    # Get the MQTT queue, clearing it if requested.
    queue_name = f"{os.environ.get('S3_USER')}/cv-job-queue"
    mb = MessageBroker()
    if args.clear_queue:
        mb.delete_queue(queue_name)
    q = mb.get_queue(queue_name)

    # Add all the jobs to the queue.
    for exp, bin_size_ms, n_states in needs_run:
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
