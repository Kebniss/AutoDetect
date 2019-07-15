import cv2
from vidaug import augmentors as va
import PIL
from PIL import Image
import math
import json
from pathlib import Path
import os
import tqdm
from torchvision.transforms import CenterCrop


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


class VideoProcessPipeline:

    def __init__(self, cv2_video, video_name="video", video_labels=None, save_every=3, scale_to_width=640):
        self.video = cv2_video
        self.video_fps = cv2_video.get(cv2.CAP_PROP_FPS)
        self.video_name = video_name.split('.')[0]
        self.preprocess = [ScaleWidth(scale_to_width).scale_width]
        self.save_every = save_every

        self.frames = []
        self.video_labels = video_labels

        success, image = cv2_video.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for p in self.preprocess:
            image = p(image)
        self.frames.append((image, 0))
        i = 1
        while success:
            if i % save_every == 0:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                for p in self.preprocess:
                    image = p(image)
                self.frames.append((image, i))
            success, image = cv2_video.read()
            i += 1
        print(f'Processed {i} frames')

    def control_fps(self, required_fps=30):
        if required_fps <= self.video_fps:
            downsample_ratio = float(required_fps) / self.video_fps
            self.video_labels = [int(l * downsample_ratio) for l in self.video_labels]
            return va.Downsample(ratio=downsample_ratio)
        else:
            upsample_ratio = float(required_fps) / self.video_fps
            self.video_labels = [int(l / upsample_ratio) for l in self.video_labels]
            return va.Upsample(ratio=upsample_ratio)

    def write_labeled_frames(self, root):
        """
        Writes labeled frames to root. Each video is saved in as frames in a folder

        named after the video name eg root/video_name/

        Images are numbered after downsampling but maintaining the original number

        eg:

        ...
        named after the video name eg root/video_name/0.jpg
        named after the video name eg root/video_name/3.jpg
        named after the video name eg root/video_name/6.jpg
        ...
        """
        write_dir = Path(root) / f"{self.video_name}"
        os.makedirs(write_dir, exist_ok=True)
        for frame, i in self.frames:
            i_str = str(i).zfill(5)
            frame.save(write_dir / f"{i_str}.png","PNG")
        return True



ROOT = "/Users/ludovica/Documents/Insight/generate_video/source/"

videos = list(Path(ROOT).glob("*.mp4"))

for p in tqdm.tqdm(videos):
    vname = p.name
    vidcap = cv2.VideoCapture(str(p))
    processed = VideoProcessPipeline(
        vidcap,
        vname,
        save_every=3,
        scale_to_width=640
    )
    processed.write_labeled_frames("/Users/ludovica/Documents/Insight/generate_video/frames/")

videos = list(Path(ROOT).glob("*.avi"))

for p in tqdm.tqdm(videos):
    vname = p.name
    vidcap = cv2.VideoCapture(str(p))
    processed = VideoProcessPipeline(
        vidcap,
        vname,
        save_every=3,
        scale_to_width=640
    )
    processed.write_labeled_frames("/Users/ludovica/Documents/Insight/generate_video/frames/")
