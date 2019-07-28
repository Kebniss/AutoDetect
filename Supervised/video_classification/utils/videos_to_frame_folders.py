"""
Given a folder of mp4, saves each of them in the FrameFolder format.

If this video's name was "abcde", this would be the structure:
root/abcde/data.csv
root/abcde/images/1.png
root/abcde/images/2.png

where data.csv contains all the columns beyond image (ie the label).
"""

import cv2
import PIL
from PIL import Image
from pathlib import Path
from torchvision.transforms import CenterCrop, Compose
from IPython.display import Image as IPythonImage
import os
from tqdm import tqdm
import json
import pandas as pd


class ToPIL(object):
    """
    Convert everything to PIL images
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if not is_PIL:
            return [PIL.Image.fromarray(img) for img in clip]


class CropOnCenter(object):
    """
    Crop image on center
    """

    def __init__(self, size):
        self.size = size
        self.crop = CenterCrop(size)

    def __call__(self, img):
        is_PIL = isinstance(img, PIL.Image.Image)
        if not is_PIL:
            img = PIL.Image.fromarray(img)
        return self.crop(img)


class ScaleWidth(object):
    """
    Scale based on width
    """

    def __init__(self, target_width, method=Image.BICUBIC):
        self.target_width = target_width
        self.method = method

    def __call__(self, img):
        return self.scale_width(img)

    def scale_width(self, img):
        is_PIL = isinstance(img, PIL.Image.Image)
        if not is_PIL:
            img = PIL.Image.fromarray(img)
        ow, oh = img.size
        if (ow == self.target_width):
            return img
        w = self.target_width
        h = int(self.target_width * oh / ow)
        return img.resize((w, h), self.method)


ROOT = Path("/Users/ludovica/Documents/Insight/data/source_data/")

with open(ROOT / "labels.json", 'r') as fin:
    labels = json.load(fin)
    labels = {
        Path(f['filename']).stem: [(t['start_frame_count'],
                                    t['end_frame_count'])
                                   for t in f['anomalies']]
        for f in labels
    }


class VideoProcessPipeline:
    def __init__(self,
                 video_path,
                 videos_labels=None,
                 new_fps=10,
                 scale_to_width=640):
        self.labeled_frames = []
        self.video_path = video_path
        cv2_video = cv2.VideoCapture(str(video_path))

        self.video = cv2_video
        self.video_fps = cv2_video.get(cv2.CAP_PROP_FPS)
        self.video_name = self.video_path.stem
        self.preprocess = Compose([
            CropOnCenter((768, 1024)),
            ScaleWidth(scale_to_width).scale_width
        ])
        if self.video_fps < 29:  # Some videos have super low fps, we just filter them out
            return
        self.save_every = self.video_fps // new_fps

        self.frames = []
        if self.video_name not in videos_labels:  # No label, no point in processing
            return

        self.frame_labels = videos_labels[self.video_name]
        self.labeled_frames = [{
            'frame_id': frame_id,
            'frame': frame,
            'label': ("positive" if any(start <= frame_id <= end
                                        for (start, end) in self.frame_labels)
                      else "negative")
        } for (frame, frame_id) in self.read_video_frames()]

    def read_video_frames(self):
        frames = []
        success, image = self.video.read()
        i = 1
        while success:
            if i % self.save_every == 0:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.preprocess(image)
                frames.append((image, i))
            success, image = self.video.read()
            i += 1
        print(f'Processed {i // self.save_every} frames')
        return frames

    def save_to_folder(self, root):
        """
        Save this video as a FrameFolder inside of root.
        If this video's name was "abcde", this would be the structure:
        root/abcde/data.csv
        root/abcde/images/1.png
        root/abcde/images/2.png

        etc
        """
        if not self.labeled_frames:
            return
        root = Path(root)
        video_root = root / self.video_name
        os.makedirs(video_root, exist_ok=True)
        os.makedirs(video_root / "images", exist_ok=True)

        for frame_dict in self.labeled_frames:
            img = frame_dict['frame']
            frame_dict['image_path'] = f"images/{frame_dict['frame_id']}.png"
            img.save(video_root / frame_dict['image_path'], "PNG")
            frame_dict.pop('frame', None)

        df = pd.DataFrame(self.labeled_frames)
        df[['frame_id', 'image_path', 'label']].to_csv(
            video_root / "data.csv", index=None)

    def as_gif(self, duration=100):
        """
        Helpful to visualize it in a notebook
        """
        self.labeled_frames[0]['frame'].save(
            "/tmp/out.gif",
            save_all=True,
            append_images=[f['frame'] for f in self.labeled_frames[1:]],
            duration=duration,
            loop=0,
        )
        return IPythonImage(filename="/tmp/out.gif")



SOURCE_ROOT = Path("/Users/ludovica/Documents/Insight/data/source_data/")

SOURCE_TRAIN_ROOT = SOURCE_ROOT / "train"
SOURCE_VALID_ROOT = SOURCE_ROOT / "validation"

TARGET_ROOT = Path("/Users/ludovica/Documents/Insight/data/frame_data/")
TARGET_TRAIN_ROOT = TARGET_ROOT / "train"
TARGET_VALID_ROOT = TARGET_ROOT / "validation"


train_video_paths = [p for p in SOURCE_TRAIN_ROOT.glob("**/*.mp4")]
validation_video_paths = [p for p in SOURCE_VALID_ROOT.glob("**/*.mp4")]


for video_path in tqdm(train_video_paths):
    if os.path.isdir(TARGET_TRAIN_ROOT / video_path.stem):
        continue  # already done, skip
    vv = VideoProcessPipeline(
        video_path, videos_labels=labels, new_fps=10, scale_to_width=640)
    vv.save_to_folder(TARGET_TRAIN_ROOT)


for video_path in tqdm(validation_video_paths):
    if os.path.isdir(TARGET_VALID_ROOT / video_path.stem):
        continue  # already done
    vv = VideoProcessPipeline(
        video_path, videos_labels=labels, new_fps=10, scale_to_width=640)
    vv.save_to_folder(TARGET_VALID_ROOT)
