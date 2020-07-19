# KITTI Human Dataset

This repository provides some tools for analyzing and 
extracting subsets from the KITTI dataset with emphasis on pedestrians
and depth data.

## Usage
### Preparing the data
- Download the depth prediction [ground truth data](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip) from KITTI's website.
- Download the corresponding raw data using `depth_val_raw_data_downloader.sh` (Only downloads the raw data corresponding to the depth validation set to save space).
- Move the contents of the validation set from ground-truth depth to `data/val/gt`.
- Restructure the raw data under `data/val/raw` so that the final structure looks like this:
```
data/
--- val/
--- --- gt/
--- --- --- 2011_09_26_drive_0002_sync/
--- --- --- 2011_09_26_drive_0005_sync/
--- --- ...
--- --- raw/
--- --- --- 2011_09_26_drive_0002_sync/
--- --- --- 2011_09_26_drive_0005_sync/
--- --- ...
```

### Extracting statistics
You can use `extract_human_stats.py` to run human detection on every frame of 'camera 2' from the raw data and save the results.

Afterwards you can use the `anayze_stats` notebook to analyze the results, and create a list of images with a minimum number of people in them (`scenes_with_min_2_people.txt` for example). 

### Dataloader
Inside `dataset.py` a pytorch dataset is defined that given a list like the one produced above, will generate RGB, ground-truth depth pairs for evaluation.
