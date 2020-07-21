from torch.utils.data import Dataset
import PIL.Image as pil
import cv2
import os.path


class KittiHumanDepthDataset(Dataset):
    def __init__(self, scenes_filename, data_root):
        scenes = []
        with open(scenes_filename) as f:
            lines = f.read().splitlines()

        for line in lines:
            scenes.append(line.split('/'))

        self.depth_files = []

        for item in scenes:
            depth_file = os.path.join(data_root, 'gt', item[0] + '_sync',
                                      'proj_depth', 'groundtruth', 'image_02',
                                      item[1])

            if os.path.isfile(depth_file):
                self.depth_files.append(depth_file)

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        depth = None
        if self.depth_files:
            depth = cv2.imread(self.depth_files[idx], -1) / 255.
        return depth

class KittiRGBDataset(Dataset):
    def __init__(self, scenes_filename, data_root):
        scenes = []
        with open(scenes_filename) as f:
            lines = f.read().splitlines()

        for line in lines:
            scenes.append(line.split('/'))

        self.rgb_files = []

        for item in scenes:
            rgb_file = os.path.join(data_root, 'raw', item[0] + '_sync',
                                    'image_02', 'data', item[1])

            if os.path.isfile(rgb_file):
                self.rgb_files.append(rgb_file)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_im = None
        if self.rgb_files:
            rgb_im = cv2.imread(self.rgb_files[idx])
        return rgb_im


