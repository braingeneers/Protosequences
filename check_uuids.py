import sys
import itertools

from hmmsupport import all_experiments, Raster


if __name__ == '__main__':

    uuids = itertools.chain(*[arg.split(',') for arg in sys.argv[1:]])

    for uuid in uuids:
        exps = all_experiments(uuid)

        for exp in exps:
            print(uuid, exp, end=' ')
            try:
                r = Raster(uuid, exp, 30)
                pr = r.coarse_rate().max()
                print(f'has {r.N} units, peak rate {pr}')
            except Exception as e:
                print(type(e), e)
