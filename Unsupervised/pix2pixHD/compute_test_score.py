### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import re
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from glob import glob
from pathlib import Path
from time import time

import pandas as pd
import json
from tqdm import tqdm
from PIL import Image
import torch
from torch import nn
import shutil
import video_utils
import image_transforms
import argparse
from data.multi_frame_dataset import MultiFrameDataset

class MeanVarOptions(TestOptions):
    def __init__(self):
        TestOptions.__init__(self)
        self.parser.add_argument('--root-dir', help='dir containing the two classes folders', dest="root_dir")
        self.parser.add_argument('--gpu', type=bool, default=False, help='Train on GPU')
        # self.parser.add_argument('--mean-var', help='path to file with mean and std from validation set')

opt = MeanVarOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# additional enforced options for video
opt.video_mode = True
opt.label_nc = 0
opt.no_instance = True
opt.resize_or_crop = "none"

# with open(Path(opt.mean_var) / 'mean_std.json', 'r') as fin:
#     ms = json.load(fin)
#     mean = float(ms['mean'])
#     std = float(ms['std'])

mean = 0.013849902898073196
std = 0.007992868311703205

model = create_model(opt)

# Not real code TODO change with opt as MultiFrameDataset wants .initialize()...
print('Processing has_target folder')
has_tgt = MultiFrameDataset()
opt.dataroot = str(Path(opt.root_dir) / "has_target")
has_tgt.initialize(opt)

rows = []
with torch.no_grad():
    for i, data in enumerate(tqdm(has_tgt)):
        cur_frame = Image.open(data['left_path'])
        next_frame = Image.open(data['right_path'])

        cur_frame = video_utils.im2tensor(cur_frame)
        next_frame = video_utils.im2tensor(next_frame)

        if opt.gpu:
            cur_frame = cur_frame.to('cuda')
            next_frame = next_frame.to('cuda')

        t0 = time()
        generated_next_frame = video_utils.next_frame_prediction(model, cur_frame)
        t1 = time()
        loss = nn.MSELoss()
        cur_loss = float(loss(generated_next_frame, next_frame))

        row = {
            'video_name': Path(cur_frame).parent.name,
            'cur_frame': Path(cur_frame).name,
            'next_frame': Path(next_frame).name,
            'MSE': cur_loss,
            'label': 0 if cur_loss < mean+2*std else 1,
            'inference_time': t1-t0,
        }
        rows.append(row)


print('Processing normal folder')
no_tgt = MultiFrameDataset()
opt.dataroot = str(Path(opt.root_dir) / "normal")
no_tgt.initialize(opt)

with torch.no_grad():
    for i, data in enumerate(no_tgt):
        cur_frame = Image.open(data['left_path'])
        next_frame = Image.open(data['right_path'])

        cur_frame = video_utils.im2tensor(cur_frame)
        next_frame = video_utils.im2tensor(next_frame)

        if opt.gpu:
            cur_frame = cur_frame.to('cuda')
            next_frame = next_frame.to('cuda')

        t0 = time()
        generated_next_frame = video_utils.next_frame_prediction(model, cur_frame)
        t1 = time()
        loss = nn.MSELoss()
        cur_loss = float(loss(generated_next_frame, next_frame))

        row = {
            'video_name': Path(cur_frame).parent.name,
            'cur_frame': Path(cur_frame).name,
            'next_frame': Path(next_frame).name,
            'MSE': cur_loss,
            'label': 0 if cur_loss < mean+2*std else 1,
            'inference_time': t1-t0,
        }
        rows.append(row)

df = pd.DataFrame(rows, columns = ['video_name', 'cur_frame', 'next_frame', 'diff', 'label', 'inference_time'])
df.to_csv('anomalies.csv', index=False)
