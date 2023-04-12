# Load HMMs for Experiment
#
# A script which takes a data source, experiment ID, bin size, and number of
# hidden states, and ensures that that HMM is in the S3 cache.
import sys
import numpy as np
from hmmsupport import cache_models

if __name__ == '__main__':
    try:
        if len(sys.argv) == 5:
            sys.argv += ['real']
        if len(sys.argv) == 6:
            sys.argv += ['ssm']
        _, source, exp, bin_size_str, nhs_str, surr, method = sys.argv

        bin_size_ms = int(bin_size_str)
        nhses = nhs_str.split('-')
        if len(nhses) == 1:
            n_stateses = [int(nhses[0])]
        else:
            nhsmin, nhsmax = nhses
            n_stateses = np.arange(int(nhsmin), int(nhsmax)+1)

    except Exception as e:
        print('Invalid arguments.', e)
        sys.exit(1)

    print(f'Caching HMMs for {source}/{exp}[{surr}] '
          f'using {bin_size_ms}ms bins')
    print('Will use K in', n_stateses)
    cache_models(source, exp, bin_size_ms, n_stateses, surr, method, True)
