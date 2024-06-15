import numpy as np
import hdf5plugin
import h5py
import argparse

is_sorted = lambda a: np.all(a[:-1] <= a[1:])

def get_ms_to_idx(all_ts):
    only = len(all_ts)
    start_idx = 0
    result = []
    done = False
    interval = int(1e8)
    while not done:
        print(start_idx)
        if start_idx + interval >= only:
            interval = only - start_idx
            done = True
        ts = all_ts[start_idx:start_idx+interval]
        frame_idxs = ts // 1000
        _, first_idx = np.unique(frame_idxs, return_index=True)
        result.extend(first_idx[:-1] + start_idx)
        start_idx += first_idx[-1]
    result.append(start_idx)
    result.append(len(all_ts))
    return result

def sort_one(dset, N, tgt, idxs):
    tgt[:] = dset[:N][idxs]

def compute(data):
    data = h5py.File(args.filtered_file, "r")
    target = h5py.File(args.target_file, "w")
    N = len(data["events/t"])
    td = target.create_dataset("events/t", (N,), dtype=np.uint32)
    pd = target.create_dataset("events/p", (N,), dtype=np.uint8)
    yd = target.create_dataset("events/y", (N,), dtype=np.uint16)
    xd = target.create_dataset("events/x", (N,), dtype=np.uint16)

    ts = data["events/t"]
    ps = data["events/p"]
    xs = data["events/x"]
    ys = data["events/y"]
    sorted_idx = np.argsort(ts[:N])
    sort_one(ts, N, td, sorted_idx)
    sort_one(ps, N, pd, sorted_idx)
    sort_one(ys, N, yd, sorted_idx)
    sort_one(xs, N, xd, sorted_idx)

    h5file = h5py.File(args.unfiltered_file, "r")
    t_offset = h5file["t_offset"]
    ms_to_idx = get_ms_to_idx(td)
    target.create_dataset("ms_to_idx", (len(ms_to_idx),), dtype=np.uint64, data=ms_to_idx)
    target.create_dataset("t_offset", t_offset.shape, dtype=t_offset.dtype, data=t_offset)
    target.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--unfiltered_file', type=str, default="sequences/unfiltered_events.h5")
    parser.add_argument('-f', '--filtered_file', type=str, default="sequences/filtered_events.h5")
    parser.add_argument('-t', '--target_file', type=str, default="sequences/sorted_events.h5")
    args = parser.parse_args()
    compute(args)