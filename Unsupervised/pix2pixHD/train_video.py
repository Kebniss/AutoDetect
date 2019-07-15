### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import torch
from torch.autograd import Variable
from data.base_dataset import get_params, get_transform
from PIL import Image

import os
import numpy as np
import shutil
import video_utils

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

# additional enforced options for video
opt.video_mode = True
opt.label_nc = 0
opt.no_instance = True

# could be changed, but not tested
opt.resize_or_crop = "none"
opt.batchSize = 1
opt.flat = False

# this debug directory will contain input/output frame pairs
if opt.debug:
    debug_dir = os.path.join(opt.checkpoints_dir, opt.name, 'debug')
    if os.path.isdir(debug_dir):
        shutil.rmtree(debug_dir)
    os.mkdir(debug_dir)

if opt.scheduled_sampling:
    print('ATTN! scheduled_sampling True by default')
    if opt.batchSize > 1:
        raise Exception('(for now) in "scheduled sampling" mode, --batchSize has to be 1')
    if not opt.serial_batches:
        raise Exception('(for now) in "scheduled sampling" mode, the --serial_batches option is necessary')
    if not opt.no_flip:
        raise Exception('(for now) in "scheduled sampling" mode, the --no_flip option is necessary')
    latest_generated_frame = None
    recursion = 0

opt.video_mode = True
# load frames dataset at the specified location
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

print(f'Training data loaded from: {opt.dataroot}')
print(f'Saving model to: {opt.name}')

dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
total_steps = (start_epoch-1) * dataset_size + epoch_iter
# print(f'Cur mem allocated: {torch.cuda.memory_allocated() / 1e6} MB')
model = create_model(opt)
# print(f'After model init. Cur mem allocated: {torch.cuda.memory_allocated() / 1e6} MB')
if opt.gpu:
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
        # print(f'Cur mem allocated: {torch.cuda.memory_allocated() / 1e6} MB')
        # move model to gpu
        model = model.to('cuda')
        # print(f'After model moved. Cur mem allocated: {torch.cuda.memory_allocated() / 1e6} MB')

visualizer = Visualizer(opt)
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        
        left_frame = Image.open(data['left_path'][0])
        right_frame = Image.open(data['right_path'][0])

        params = get_params(opt, left_frame.size)
        transform = get_transform(opt, params)

        left_frame = transform(left_frame.convert('RGB'))
        right_frame = transform(right_frame.convert('RGB'))

        if opt.gpu:
            left_frame = left_frame.unsqueeze(0).to('cuda')
            # print(f'After left frame moved. Cur mem allocated: {torch.cuda.memory_allocated() / 1e6} MB')
            right_frame = right_frame.unsqueeze(0).to('cuda')
            # print(f'After right frame moved. Cur mem allocated: {torch.cuda.memory_allocated() / 1e6} MB')

        if opt.debug:
            video_utils.save_tensor(left_frame, debug_dir + "/step-%d-left-r%d.jpg" % (total_steps, recursion))
            video_utils.save_tensor(right_frame, debug_dir + "/step-%d-right.jpg" % total_steps)

        # print(f"LEFT FRAME: {left_frame.size()}")
        # print(f"RIGHT FRAME: {right_frame.size()}")

        losses, latest_generated_frame = model(
            left_frame, None,
            right_frame, None,
            infer=opt.scheduled_sampling
        )

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

        ############### Backward Pass - frame1->frame2 ####################
        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()

        # update discriminator weights
        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D.step()

        #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
