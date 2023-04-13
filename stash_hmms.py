# Load HMMs for Experiment
#
# A script which takes a data source, experiment ID, bin size, and number of
# hidden states, and ensures that that HMM is in the S3 cache.
import sys
import numpy as np
from hmmsupport import cache_models


def maybe_list(str_str):
    if ',' in str_str:
        return str_str.split(',')
    else:
        return str_str


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
    try:
        if len(sys.argv) == 5:
            sys.argv += ['real']
        if len(sys.argv) == 6:
            sys.argv += ['ssm']
        _, source, exp, bin_size_str, nhs_str, surr, method = sys.argv

        bin_sizes = parse_range_str(bin_size_str)
        n_stateses = parse_range_str(nhs_str)

    except Exception as e:
        print('Invalid arguments.', e)
        sys.exit(1)

    print(f'Caching HMMs for {source}/{exp}[{surr}] using `{method}`.')
    print('Will use K in', n_stateses)
    print('Will use T in', bin_sizes)
    cache_models(source, maybe_list(exp), bin_sizes, n_stateses,
                 maybe_list(surr), method, True)
