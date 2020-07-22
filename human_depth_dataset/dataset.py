import cv2
import pickle
import os.path
import numpy as np
import PIL.Image as pil
from torch.utils.data import Dataset


class KittiHumanDepthDataset(Dataset):
    def __init__(self, scenes_filename, data_root='../data/kitti/val/', mask_file='../data/kitti/yolact.pkl'):
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
        
        self.masks = None
        if mask_file != None:
            with open(mask_file, 'rb') as f:
                self.masks = pickle.load(f)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_im = pil.open(self.rgb_files[idx])

        depth = cv2.imread(self.depth_files[idx], -1) / 255.

        file = self.rgb_files[idx].split('/')[-4] + '/' + self.rgb_files[idx].split('/')[-1]
        mask = self.masks[file] if self.masks != None else None
        return {'rgb': rgb_im, 'depth': depth, 'index': file, 'mask': mask}


class RGBDPeopleDataset(Dataset):
    def __init__(self, data_root='../data/rgbd/', mask_file='../data/rgbd/yolact.pkl'):
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
            
        self.masks = None
        if mask_file != None:
            with open(mask_file, 'rb') as f:
                self.masks = pickle.load(f)

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
        
        index = self.rgb_files[idx].split('/')[-1]
        mask = self.masks[index] if self.masks != None else None

        return {'rgb': rgb_im, 'depth': depth, 'index': index, 'mask': mask}
