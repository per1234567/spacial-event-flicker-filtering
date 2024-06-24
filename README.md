# Improving Optical Flow Estimation Accuracy Using Space-Aware De-Flickering

This repository is part of a 2024 [Research Project](https://github.com/TU-Delft-CSE/Research-Project) conducted at TU Delft.

This code contains the algorithm discussed in the paper. Additional files aiding in visualization and reproduction are included as well.

## Setup

1. Python `3.11.7` is known to work. Download the dependencies:

```
$ pip install -r requirements.txt
```

Use of Conda is recommended.

## Minimal example

1. Download an events file (`events_left.zip`) from the [DSEC repository](https://dsec.ifi.uzh.ch/dsec-datasets/download/). Extract the file `events.h5` to the location `sequences/events.h5`.

2. Run script that filters the event set.

```
$ python main.py
```

3. Run the script that visualizes the events as a `.mp4` file.

```
$ python visualize.py
```

## Extra files

We share other files necessarily modified to facilitate benchmarking, relevant to other projects.

### TamingCM

Files modified from the official [Taming Contrast Maximization repository](https://github.com/tudelft/taming_event_flow).

#### h5.py

This is the dataloader file, it was modified to allow compatibility with event sets in the same format as downloaded from the DSEC repository. It contains example filepaths for the non-test sequence `zurich_city_10_a`. To use this file, set up TamingCM as described by its authors, then replace the file `dataloader/h5.py` with this one.

### EFR

Files modified from the official [EFR repository](https://github.com/ziweiWWANG/EFR). We additionally use the [HighFive](https://vcpkg.io/en/package/highfive) package.

#### repackage.py

This file is used to reformat a DSEC dataset to make it compatible with our h5 dataloader for EFR.

#### main.cpp

The `main.cpp` file modified to load events from an h5 file. To use this file, set up EFR as described by its authors, then replace the file `main.cpp` with this one.

#### comb_filer.cpp

The `comb_filter.cpp` file modified to write events into an h5 file. To use this file, set up EFR as described by its authors, then replace the file `comb_filter.cpp` with this one.

#### sort_events.py

This file is used to sort events outputted by the writer in `comb_filter.cpp` by timestamp.