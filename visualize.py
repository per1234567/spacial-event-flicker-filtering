import numpy as np
import hdf5plugin
import h5py
import cv2
import argparse
import dsec_config as config

def get_first_idxs(fps, event_path, first_n):
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
        frame_idxs = ts // (1000000 // fps)
        _, first_idx = np.unique(frame_idxs, return_index=True)
        if len(first_idx) == 1:
            return result
        result.extend(first_idx[:-1] + start_idx)
        start_idx += first_idx[-1]
    return result

def visualize_events(args):
    data = h5py.File(args.event_file)
    fps = args.frames_per_second
    first_idx = get_first_idxs(fps, args.event_file, args.first_n)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(args.video_file, fourcc, fps, (config.width, config.height))

    r = np.array([0, 0, 255], dtype=np.uint8)
    b = np.array([255, 0, 0], dtype=np.uint8)
    
    for s, e in zip(first_idx[:-1], first_idx[1:]):
        img = np.full((config.height, config.width, 3), 255, dtype=np.uint8)
        ps = data["events"]["p"][s:e]
        ys = data["events"]["y"][s:e]
        xs = data["events"]["x"][s:e]
        colors = np.zeros((len(ps), 3), dtype=np.uint8)
        colors[ps == 1] = b
        colors[ps == 0] = r
        img[ys, xs] = colors
        output_video.write(img)
    
    output_video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--event_file', type=str, default="sequences/filtered_events.h5")
    parser.add_argument('-v', '--video_file', type=str, default="visualization.mp4")
    parser.add_argument('-f', '--frames_per_second', type=int, default=200)
    parser.add_argument('-n', '--first_n', type=int, default=int(1e10), help="Upper bound of how many events to consider")
    args = parser.parse_args()
    visualize_events(args)