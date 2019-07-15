import os
import cv2
import argparse
from utils import *
from tqdm import tqdm
from glob import glob
from pathlib import Path


def _extract_frames(video_path, parent, start=0, sampling_f=1):
    vidcap = cv2.VideoCapture(video_path)
    success, image = success, image = vidcap.read()
    count = -1
    saved = 0
    print(f'Processing: {video_path}')
    while success:
        count += 1
        if count % 300 == 0:
            print('Processing frame: ', count)
        if count % sampling_f == 0:
            # sampling
            cv2.imwrite(''.join([dest_folder, f"/{count + start}.jpg"]), image)
            saved += 1
        success, image = vidcap.read()  # read next


    print(f'Successfully saved {saved} frames to {dest_folder}')
    return count + start


parser = argparse.ArgumentParser(
    description='build a "frame dataset" from a given video')
parser.add_argument('-input', dest="input", required=True,
    help='''Path to a single video or a folder. If path to folder the algorithm
         will extract frames from all files with extension defined in
         --extension and save them under separate folders under dest_folder.
         The frames from each video will be saved under a folder with its name.
         ''')
parser.add_argument('--dest-folder', dest="dest_folder", default='./dataset/',
    help='''Path where to store frames. NB all files in this folder will be
         removed before adding the new frames''')
parser.add_argument('--same-folder', dest="same_folder", default=False,
    help='''Set it to True if you want to save the frames of all videos to the
    same folder in ascending order going from the first frame of the first video
    to the last frame of the last video. If True frames will be saved in
    dest_folder/frames.''')
parser.add_argument('--sampling', help='how many fps', default='3')
parser.add_argument('--run-type', help='train or test', default='train')
parser.add_argument('--extension', help='avi, mp4, mov...', default='mp4')
parser.add_argument('-width', help='output width', default=640, type=int)
parser.add_argument('-height', help='output height', default=480, type=int)
args = parser.parse_args()

mkdir(args.dest_folder)

if (args.width % 32 != 0) or (args.height % 32 != 0):
    raise Exception("Please use width and height that are divisible by 32")
if os.path.isdir(args.input):
    inp = str(Path(args.input) / f'*.{args.extension}')
    videos = [v for v in glob(inp)]
    if not videos:
        raise Exception(f'No {args.extension} files in input directory {args.input}')
elif os.path.isfile(args.input):
    _, ext = get_filename_extension(args.input)
    if ext != args.extension:
        raise ValueError(f'Correct inputs: folder or path to {args.extension} file only')
    videos = [args.input]
else:
    raise ValueError(f'Correct inputs: folder or path to {args.extension} file only')

if args.same_folder:
    start = 0
    dest_folder = str(Path(args.dest_folder) / f'{args.run_type}_frames')
    mkdir(dest_folder)

for v in tqdm(videos):
    if not args.same_folder:
        start = 0
        name, _ = get_filename_extension(v)
        dest_folder = str(Path(args.dest_folder) / name)
        mkdir(dest_folder)

    start = _extract_frames(v, dest_folder, start, sampling_f=int(args.sampling))
