### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import re
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from glob import glob
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import torch
import shutil
import video_utils
import image_transforms
from data.multi_frame_dataset import MultiFrameDataset


class MeanVarOptions(TestOptions):
    def __init__(self):
        TestOptions.__init__(self)
        self.parser.add_argument('--gpu', type=bool, default=False, help='Train on GPU')
        self.parser.add_argument('--flat', type=bool, default=True, help='Flat folder structure')

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
opt.checkpoints_dir = '/home/ubuntu/pix2pixHD/checkpoints/'

# this directory will contain the generated videos
output_dir = os.path.join(opt.checkpoints_dir, opt.name, 'output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# this directory will contain the frames to build the video
frame_dir = os.path.join(opt.checkpoints_dir, opt.name, 'frames')
if os.path.isdir(frame_dir):
    shutil.rmtree(frame_dir)
os.mkdir(frame_dir)

model = create_model(opt)

positive_ds = MultiFrameDataset()
positive_ds.initialize(opt)

frame_index = 1
print(f'LEN {len(positive_ds)}')
with torch.no_grad():
    for i, data in enumerate(tqdm(positive_ds)):
        cur_frame = Image.open(data['left_path'])
        next_frame = Image.open(data['right_path'])

        cur_frame = video_utils.im2tensor(cur_frame)
        next_frame = video_utils.im2tensor(next_frame)

        if opt.gpu:
            cur_frame = cur_frame.to('cuda')
            next_frame = next_frame.to('cuda')

        generated_next_frame = video_utils.next_frame_prediction(model, cur_frame)
        
        video_utils.save_tensor(
        generated_next_frame,
        frame_dir + "/frame-%s.png" % str(frame_index).zfill(5),
        )
        frame_index += 1


print('Finished generating images')
duration_s = frame_index / opt.fps
video_id = "epoch-%s_%s_%.1f-s_%.1f-fps" % (
    str(opt.which_epoch),
    opt.name,
    duration_s,
    opt.fps
)

print(f'created video id {video_id}')
video_path = output_dir + "/" + video_id + ".mp4"
while os.path.isfile(video_path):
    video_path = video_path[:-4] + "-.mp4"
print(f'modified video path {video_path}')
video_utils.video_from_frame_directory(
    frame_dir,
    video_path,
    framerate=opt.fps,
    crop_to_720p=False,
    reverse=False
)
print('saved video')
print("video ready:\n%s" % video_path)
