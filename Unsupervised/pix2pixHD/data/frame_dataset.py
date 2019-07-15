### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, normalize
from data.image_folder import make_dataset
from functools import partial
import re


def extract_frame_id_int(path, pattern):
    # extract frame id as int
    return int(pattern.search(path).group(1))


class FrameDataset(BaseDataset):
    def __init__(self, root=None, opt=None):
        if root:
            self.root = root
            self.opt = opt
            self.dir_frames = root
            self.sort_key = partial(extract_frame_id_int,
                                    pattern=re.compile('(\d+)[\.jpg|\.png]'))
            self.frame_paths = sorted([i for i in make_dataset(self.root)],
                                      key=self.sort_key)
            self.frame_count = len(self.frame_paths)
            self.dataset_size = self.frame_count - 1

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### frames
        dir_frames = '_frames'
        self.dir_frames = os.path.join(opt.dataroot, opt.phase + dir_frames)
        self.dir_frames = self.sort()
        self.frame_paths = sorted([i for i in make_dataset(self.dir_frames)],
                                  key=self.sort_key)
        self.frame_count = len(self.frame_paths)
        self.dataset_size = self.frame_count - 1

        print("FrameDataset initialized from: %s" % self.dir_frames)
        print("contains %d frames, %d consecutive pairs" % (self.frame_count, self.dataset_size))

    def __getitem__(self, index):

        left_frame_path = self.frame_paths[index]
        right_frame_path = self.frame_paths[index+1]

        input_dict = {
            'left_path': left_frame_path,
            'right_path': right_frame_path,
        }

        return input_dict

    def __len__(self):
        # batchSize>1 not tested
        return self.dataset_size

    def name(self):
        return 'FrameDataset'
