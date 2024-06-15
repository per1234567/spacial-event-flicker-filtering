import numpy as np
import hdf5plugin
import h5py
import argparse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import dsec_config as config

tiles_w = config.width // config.tile_width
tiles_h = config.height // config.tile_height

def get_first_idxs(event_path, first_n):
    #
    # Returns a slice whose i-th index is the index of the first event in the i-th frame
    #
    data = h5py.File(event_path)
    ts = data["events"]["t"]
    only = min(first_n, len(ts))
    start_idx = 0
    result = []
    done = False
    interval = int(1e8)
    while not done:
        print(start_idx)
        if start_idx + interval >= only:
            interval = only - start_idx
            done = True
        ts = data["events"]["t"][start_idx:start_idx+interval]
        frame_idxs = ts // (1000000 // config.frames_per_second)
        _, first_idx = np.unique(frame_idxs, return_index=True)
        result.extend(first_idx[:-1] + start_idx)
        start_idx += first_idx[-1]
    result.append(start_idx)
    return result

def get_ms_to_idx(all_ts):
    #
    # Computes the ms_to_idx object using the filtered events, as available by default in the DSEC sequences
    #
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

def compute_balances(frame_idxs, ts, ps, ys, xs):
    #
    # Compute the count of +1 or -1 events that must be deleted at each frame
    #
    colors = np.zeros((len(frame_idxs)-1, config.height, config.width), dtype=np.int16)
    frame_idx = 0
    for s, e in zip(frame_idxs[:-1], frame_idxs[1:]):
        s -= frame_idxs[0]
        e -= frame_idxs[0]
        frame_ps = ps[s:e]
        frame_ys = ys[s:e]
        frame_xs = xs[s:e]
        np.add.at(colors[frame_idx], (frame_ys, frame_xs), np.where(frame_ps == 1, 1, -1))
        frame_idx += 1

    subtile_balances = np.ndarray((frame_idx, tiles_h, tiles_w), dtype=np.float64)
    for h in range(0, config.height-1, config.tile_height):
        for w in range(0, config.width-1, config.tile_width):
            color_slice = colors[:, h:h+config.tile_height, w:w+config.tile_width]
            color_balance_series = np.sum(color_slice, axis=(1, 2))
            expected_balance = gaussian_filter(color_balance_series, 1)
            balance_changes = color_balance_series - expected_balance
            subtile_balances[:, h // config.tile_height, w // config.tile_width] = balance_changes

    return subtile_balances

def filter_with_balances(frame_idxs, ts, ps, ys, xs, subtile_balances):
    #
    # Filter a cell using the computed balances
    #
    events = np.ndarray((len(ts), 5), dtype=np.uint32)
    events_idx = 0
    for frame_idx, (s, e) in enumerate(zip(frame_idxs[:-1], frame_idxs[1:])):
        s -= frame_idxs[0]
        e -= frame_idxs[0]
        frame_ts = ts[s:e]
        frame_ps = ps[s:e]
        frame_ys = ys[s:e]
        frame_xs = xs[s:e]
        balance_idxs = frame_idx * (tiles_w * tiles_h) + (frame_ys.astype(np.uint32) // config.tile_height) * tiles_w + (frame_xs.astype(np.uint32) // config.tile_width)
        stacked = np.stack((balance_idxs, frame_ts, frame_ps, frame_ys, frame_xs), axis=-1, dtype=np.uint32)
        events[events_idx:events_idx+len(stacked)] = stacked
        events_idx += len(stacked)

    sorted_idxs = np.argsort(events[:,0])
    events = events[sorted_idxs]
    events_grouped = np.split(events, np.unique(events[:,0], return_index=True)[1][1:])
    balances_flat = subtile_balances.flatten().astype(np.int32)
    events_filtered = []
    for group in tqdm(events_grouped):
        bal = balances_flat[group[0][0]]
        if bal == 0:
            events_filtered.extend(group)
            continue
        ups = group[np.where(group[:,2] == 1)]
        downs = group[np.where(group[:,2] == 0)]
        np.random.shuffle(ups)
        np.random.shuffle(downs)

        if bal > 0:
            up_map = {}
            for event in ups:
                key = (event[3] // 2) + (event[4] // 2) * config.height
                if key not in up_map:
                    up_map[key] = []
                up_map[key].append(event)
            while bal > 0 and len(up_map) > 0:
                to_delete = []
                for k, events in up_map.items():
                    events.pop()
                    bal -= 1
                    if bal == 0:
                        break
                    if len(events) == 0:
                        to_delete.append(k)
                for k in to_delete:
                    del up_map[k]
            events_filtered.extend(downs)
            for events in up_map.values():
                events_filtered.extend(events)
        else:
            down_map = {}
            for event in downs:
                key = (event[3] // 2) + (event[4] // 2) * config.height
                if key not in down_map:
                    down_map[key] = []
                down_map[key].append(event)
            while bal < 0 and len(down_map) > 0:
                to_delete = []
                for k, events in down_map.items():
                    events.pop()
                    bal += 1
                    if bal == 0:
                        break
                    if len(events) == 0:
                        to_delete.append(k)
                for k in to_delete:
                    del down_map[k]
            events_filtered.extend(ups)
            for events in down_map.values():
                events_filtered.extend(events)
        
    events_filtered = np.array(events_filtered, dtype=np.uint32)
    events_filtered = events_filtered[np.argsort(events_filtered[:, 1])]
    print(events_filtered.shape)

    return events_filtered

def compute(events_path, dest_path, first_n):
    #
    # Filters the event set
    #
    h5file = h5py.File(events_path)
    data = h5file["events"]
    first_idxs_of_frame = get_first_idxs(events_path, first_n)

    f = h5py.File(dest_path, "w")
    chunk_size=int(5e6)
    td = f.create_dataset("events/t", (0,), dtype=np.uint32, maxshape=(None,), chunks=(chunk_size,))
    pd = f.create_dataset("events/p", (0,), dtype=np.uint8, maxshape=(None,), chunks=(chunk_size,))
    yd = f.create_dataset("events/y", (0,), dtype=np.uint16, maxshape=(None,), chunks=(chunk_size,))
    xd = f.create_dataset("events/x", (0,), dtype=np.uint16, maxshape=(None,), chunks=(chunk_size,))

    frames_per_tile = int(config.frames_per_second * config.tile_duration)
    cell_starts = range(0, len(first_idxs_of_frame)-frames_per_tile, frames_per_tile)
    for i in tqdm(cell_starts):
        s = first_idxs_of_frame[i]
        e = first_idxs_of_frame[i + frames_per_tile]
        ts = data["t"][s:e]
        ps = data["p"][s:e]
        ys = data["y"][s:e]
        xs = data["x"][s:e]
        frame_idxs = first_idxs_of_frame[i:i+frames_per_tile+1]
        center_balances = compute_balances(frame_idxs, ts, ps, ys, xs)
        all_events = filter_with_balances(frame_idxs, ts, ps, ys, xs, center_balances)
        for dset, idx in zip([td, pd, yd, xd], [1,2,3,4]):
            dset.resize(dset.shape[0] + len(all_events), axis=0)
            dset[-len(all_events):] = all_events[:,idx]
    
    # Handle the last few events that do not perfectly fit into a cell
    i = cell_starts[-1] + frames_per_tile
    s = first_idxs_of_frame[-frames_per_tile]
    e = first_idxs_of_frame[-1]
    ts = data["t"][s:e]
    ps = data["p"][s:e]
    ys = data["y"][s:e]
    xs = data["x"][s:e]
    frame_idxs = first_idxs_of_frame[-frames_per_tile:]
    frame_idxs = np.append(frame_idxs, -1)
    center_balances = compute_balances(frame_idxs, ts, ps, ys, xs)
    all_events = filter_with_balances(frame_idxs, ts, ps, ys, xs, center_balances)
    last_t = data["t"][first_idxs_of_frame[cell_starts[-1] + frames_per_tile]]
    all_events = all_events[np.where(all_events[:,1] >= last_t)]
    for dset, idx in zip([td, pd, yd, xd], [1,2,3,4]):
        dset.resize(dset.shape[0] + len(all_events), axis=0)
        dset[-len(all_events):] = all_events[:,idx]

    t_offset = h5file["t_offset"]
    ms_to_idx = get_ms_to_idx(td)
    f.create_dataset("ms_to_idx", (len(ms_to_idx),), dtype=np.uint64, data=ms_to_idx)
    f.create_dataset("t_offset", t_offset.shape, dtype=t_offset.dtype, data=t_offset)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', type=str, default="sequences/events.h5")
    parser.add_argument('-t', '--target_file', type=str, default="sequences/filtered_events.h5")
    parser.add_argument('-n', '--first_n', type=int, default=int(1e10), help="Upper bound of how many events to consider")
    args = parser.parse_args()
    compute(args.source_file, args.target_file, args.first_n)