# Fit HMMs
#
# A script which takes a data source plus lists of experiment IDs, bin
# sizes, and numbers of hidden states, checks which of those models are not
# cached, and either adds them to the HMM job queue or just runs them
# locally, depending whether the option --local is provided.
import sys
import itertools
import numpy as np
import hmmsupport
from braingeneers.iot.messaging import MessageBroker


def ensure_list(str_str):
    if ',' in str_str:
        return str_str.split(',')
    else:
        return [str_str]


def parse_range_str(range_str):
    if ',' in range_str:
        return np.array([int(x) for x in range_str.split(',')])
    elif ':' in range_str:
        parts = range_str.split(':', 2)
        step = int(parts[2]) if len(parts) == 3 else None
        return np.arange(int(parts[0]), int(parts[1])+1, step)
    elif '-' in range_str:
        start, end = range_str.split('-')
        return np.arange(int(start), int(end)+1)
    else:
        return np.array([int(range_str)])


if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    if dryrun:
        sys.argv.remove('--dryrun')

    local = '--local' in sys.argv
    if local:
        sys.argv.remove('--local')

    # Validate the arguments.
    try:
        if len(sys.argv) == 5:
            sys.argv.append('real')
        if len(sys.argv) == 6:
            sys.argv.append('default')
        _, source, exp, bin_size_str, nhs_str, surr, library = sys.argv

        bin_sizes = parse_range_str(bin_size_str)
        n_stateses = parse_range_str(nhs_str)

    except Exception as e:
        print('Invalid arguments.', e)
        sys.exit(1)

    if library not in hmmsupport._HMM_METHODS:
        print(f'Invalid library `{library}`.')
        sys.exit(1)


    # Verbosely print the full parameter set.
    if exp == '*':
        exps = hmmsupport.all_experiments(source)
    else:
        exps = ensure_list(exp)
    surrs = ensure_list(surr)
    print(f'Fitting HMMs using {library} for experiments:')
    for exp, surr in itertools.product(exps, surrs):
        print(f'  {source}/{exp}[{surr}]')
    print('Will use K in', n_stateses)
    print('Will use T in', bin_sizes)


    # Check which parameters actually need re-run.
    needs_run = [
        p for p in itertools.product(exps, bin_sizes, n_stateses, surrs)
        if not hmmsupport.is_cached(source, *p, library=library)
    ]
    if not needs_run:
        print('All HMMs are already cached.')
        sys.exit(0)

    print(f'{len(needs_run)} HMMs need to be fit:')
    for exp, T, K, surrogate in needs_run:
        print(f'  {source}/{exp}[{surrogate}] with {T=}ms, {K=}.')

    # If this is a dry run, don't actually bother queueing anything.
    if dryrun:
        sys.exit()

    # If local, don't use MQTT, just do all the fits in a loop here.
    elif local:
        for exp, bin_size_ms, n_states, surrogate in needs_run:
            hmmsupport.get_fitted_hmm(source, exp, bin_size_ms, n_states,
                                      surrogate, library, verbose=True)
    
    # Get the MQTT queue and add all the jobs to it.
    else:
        q = MessageBroker().get_queue('atspaeth/hmm-job-queue')
        for exp, bin_size_ms, n_states, surrogate in needs_run:
            q.put(dict(
                retries_allowed=3,
                params=dict(source=source, exp=exp, bin_size_ms=bin_size_ms,
                            n_states=n_states, surrogate=surrogate,
                            library=library)))
