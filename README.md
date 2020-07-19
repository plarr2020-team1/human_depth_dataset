# KITTI Human Dataset

This repository provides some tools for analyzing and 
extracting subsets from the KITTI dataset with emphasis on pedestrians
and depth data.

## Usage
### Preparing the data
- Download the depth prediction ground truth data from KITTI's website
- Download the corresponding raw data using `depth_val_raw_data_downloader.sh`
- Move the contents of the validation set from ground-truth depth to `data/val/gt`.
- Restructure the raw data under `data/val/raw` to look like this:
```
data/
--- gt/
--- --- 2011_09_26_drive_0002_sync/
--- --- 2011_09_26_drive_0005_sync/
--- ...
--- raw/
--- --- 2011_09_26_drive_0002_sync/
--- --- 2011_09_26_drive_0005_sync/
--- ...
```