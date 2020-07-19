from torch.utils.data import Dataset
import PIL.Image as pil
import cv2
import os.path

data_root = "./data/val/"


class KittiHumanDepthDataset(Dataset):
    def __init__(self, scenes_filename):
        self.scenes = []
        with open(scenes_filename) as f:
            scenes = f.read().splitlines()

        for scene in scenes:
            self.scenes.append(scene.split('/'))

    def __getitem__(self, idx):
        rgb_file = os.path.join(data_root, 'raw',
                                self.scenes[idx][0] + '_sync',
                                'image_02', 'data', self.scenes[idx][1])

        depth_file = os.path.join(data_root, 'gt',
                                  self.scenes[idx][0] + '_sync', 'proj_depth',
                                  'groundtruth', 'image_02',
                                  self.scenes[idx][1])
        rgb_im = pil.open(rgb_file)

        depth = cv2.imread(depth_file, -1) / 255.

        return rgb_im, depth
