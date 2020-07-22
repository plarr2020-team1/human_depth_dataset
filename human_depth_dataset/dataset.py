from torch.utils.data import Dataset
import cv2
import os.path
import PIL.Image as pil
import numpy as np


class KittiHumanDepthDataset(Dataset):
    def __init__(self, scenes_filename, data_root='../data/kitti/val/'):
        scenes = []
        with open(scenes_filename) as f:
            lines = f.read().splitlines()

        for line in lines:
            scenes.append(line.split('/'))

        self.rgb_files = []
        self.depth_files = []

        for item in scenes:
            rgb_file = os.path.join(data_root, 'raw', item[0] + '_sync',
                                    'image_02', 'data', item[1])
            depth_file = os.path.join(data_root, 'gt', item[0] + '_sync',
                                      'proj_depth', 'groundtruth', 'image_02',
                                      item[1])

            if os.path.isfile(rgb_file) and os.path.isfile(depth_file):
                self.rgb_files.append(rgb_file)
                self.depth_files.append(depth_file)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):

        rgb_im = cv2.imread(self.rgb_files[idx])

        depth = cv2.imread(self.depth_files[idx], -1) / 255.

        return rgb_im, depth


class RGBDPeopleDataset(Dataset):
    def __init__(self, data_root='../data/rgbd/'):
        rgb_dir = 'rgb'
        depth_dir = 'depth'

        self.rgb_files = [os.path.join(data_root, rgb_dir, filename) for
                          filename in
                          os.listdir(os.path.join(data_root, rgb_dir))]

        self.depth_files = []
        for path in self.rgb_files:
            filename = os.path.splitext(os.path.basename(path))[0]
            self.depth_files.append(
                os.path.join(data_root, depth_dir, filename+'.pgm'))

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_np = np.rot90(cv2.imread(self.rgb_files[idx]))
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        rgb_im = pil.fromarray(rgb_np)

        depth = cv2.imread(self.depth_files[idx], -1).newbyteorder()

        # According to the dataset paper: http://www2.informatik.uni-freiburg.de/~spinello/spinelloIROS11.pdf
        depth = 8 * 0.075 * 594.2 / (1084 - depth)
        depth = np.rot90(depth)

        return rgb_im, depth
