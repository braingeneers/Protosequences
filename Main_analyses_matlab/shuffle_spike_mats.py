import numpy as np
import pickle
from scipy.io import loadmat, savemat
import h5py
from tqdm import tqdm
from pathlib import Path
from random import seed

NUM_DATASETS = 100
SWAP_PER_SPIKE = 5
NUM_SEC = 180
SEED = [1,2,3,4,5,6,7,8,9,10]
num_shuff_datasets = int(np.ceil(NUM_DATASETS/len(SEED)))

ALL_FILES = [
    "PATH1/TO/SORTED/SPIKE/MATRIX",
    "PATH2/TO/SORTED/SPIKE/MATRIX",
]


def swap(ar, idxs):
    idx0 = np.random.randint(len(idxs[0]))
    idx1 = np.random.randint(len(idxs[0]))
    i0, j0 = idxs[0][idx0], idxs[1][idx0]
    i1, j1 = idxs[0][idx1], idxs[1][idx1]
    if i0 == i1 or j0 == j1 or ar[i0, j1] == 1.0 or ar[i1, j0] == 1.0:
        return False
    ar[i0, j0] = ar[i1, j1] = 0.0
    ar[i0, j1] = ar[i1, j0] = 1.0
    idxs[0][idx0], idxs[1][idx0] = i0, j1
    idxs[0][idx1], idxs[1][idx1] = i1, j0
    return True

def randomize(ar, swap_per_spike=SWAP_PER_SPIKE):
    ar = ar.copy()
    idxs = np.where(ar == 1.0)
    cnt_swap = 0
    for _ in range(int((swap_per_spike+1) * np.sum(ar))):
        if swap(ar, idxs):
            cnt_swap += 1

    if cnt_swap < swap_per_spike * np.sum(ar):
        for _ in range(int((swap_per_spike+1) * np.sum(ar))):
            if swap(ar, idxs):
                cnt_swap += 1

    if cnt_swap < swap_per_spike * np.sum(ar):
        print("ERROR: Not sufficient succesfull swaps, only {} of {} required".format(cnt_swap, swap_per_spike * np.sum(ar)))

    return ar



from multiprocessing import Pool

def process(args):
    fn, s = args
    seed(s)

    try:
        f = loadmat(f"{fn}.mat")
        ar = np.array(f['t_spk_mat']).T.astype(np.float32)
    except:
        f = h5py.File(f"{fn}.mat")
        ar = np.array(f['t_spk_mat']).astype(np.float32)

    ar = ar[:, :NUM_SEC * 1000]
    res = [randomize(ar) for _ in tqdm(range(num_shuff_datasets))]

    res_dict = {"all_shuff":res}

    savemat("{}_shuff_s{}.mat".format(fn,s), res_dict)



if __name__ == "__main__":
    tasks = []
    for fn in ALL_FILES:
        for s in SEED:
            tasks.append((fn, s))

    for task in tasks:
        process(task)
