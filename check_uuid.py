from hmmsupport import all_experiments, Raster

if __name__ == '__main__':

    import sys
    uuid = sys.argv[1]

    exps = all_experiments(uuid)

    for exp in exps:
        try:
            r = Raster(uuid, exp, 30)
            pr = r.coarse_rate().max()
            print(f'{exp} has {r.N} units, peak rate {pr}')
        except Exception as e:
            print(e)

