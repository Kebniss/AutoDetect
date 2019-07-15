import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from data.frame_dataset import FrameDataset
from typing import List
import os
from PIL import Image
from tqdm import tqdm


class MultiFrameDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dirs = [
            os.path.join(self.root, o) for o in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, o))
        ]

        self.datasets = []
        for d in tqdm(self.dirs):
            self.datasets.append(FrameDataset(d, self.opt))
        self.data = []
        for ds in self.datasets:
            self.data.extend((ds[i]) for i in range(len(ds)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
