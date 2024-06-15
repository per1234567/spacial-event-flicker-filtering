import numpy as np
import hdf5plugin
import h5py
import argparse
from tqdm import tqdm

def compute(source_file, target_file, first_n):
    data = h5py.File(source_file)
    ts = data["events"]["t"]
    ps = data["events"]["p"]
    ys = data["events"]["y"]
    xs = data["events"]["x"]

    N = min(first_n, len(ts))
    chunk_size = int(1e6)

    f = h5py.File(target_file, "w")
    td = f.create_dataset("events/t", (0,), dtype=np.uint32, maxshape=(None,), chunks=(chunk_size,))
    pd = f.create_dataset("events/p", (0,), dtype=np.uint8, maxshape=(None,), chunks=(chunk_size,))
    yd = f.create_dataset("events/y", (0,), dtype=np.uint16, maxshape=(None,), chunks=(chunk_size,))
    xd = f.create_dataset("events/x", (0,), dtype=np.uint16, maxshape=(None,), chunks=(chunk_size,))

    for i in tqdm(range(0, N, chunk_size)):
        for og, proc in zip([ts, ps, ys, xs], [td, pd, yd, xd]):
            dt = og[i:i+chunk_size]
            proc.resize(proc.shape[0] + len(dt), axis=0)
            proc[-len(dt):] = dt

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', type=str, default="sequences/events.h5")
    parser.add_argument('-t', '--target_file', type=str, default="sequences/repackaged_events.h5")
    parser.add_argument('-n', '--first_n', type=int, default=int(1e10), help="Upper bound of how many events to consider")
    args = parser.parse_args()
    compute(args.source_file, args.target_file, args.first_n)