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
        self.preprocess = [ScaleWidth(scale_to_width).scale_width] #CropOnCenter((768, 1024)),
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

    def label_frames(self, chunk_size=30, drop_last=False):
        """
        Creates labeled frames.

        @params
        - chunk_size: how long each list of frames is
        - drop_last: if the last chunk is of len < chunk_size, choose whether to drop it (defaults to False)

        @returns List[Dict]. Each dict has these keys:
        - 'label': one of ['has_anomaly', 'normal']
        - 'frames': list of len chunk_size, each containing one PIL.Image
        """
        self.chunk_size = chunk_size

        chunks = [
            self.frames[x:x+chunk_size]
            for x in range(0, len(self.frames), chunk_size)
        ]
        anormal_chunks = {math.floor(lab/chunk_size/self.save_every) for lab in self.video_labels}
        self.frames = [
            {
                'label': 'has_anomaly' if i in anormal_chunks else 'normal',
                'frames': c,
            }
            for i, c in enumerate(chunks)
        ]
        if len(self.frames[-1]['frames']) < chunk_size and drop_last:
            self.frames.pop()

    def write_labeled_frames(self, root):
        """
        Writes labeled frames to root. Uses the label as a subfolder, then writes all chunks into the folder.

        FOLDER STRUCTURE:

        root/has_anomaly
        root/normal

        then inside you would have all the chunks, using self.video_name + f"_chunk{i}" eg

        root/has_anomaly/anomaly_2019-06-11_15-12-00_chunk0/

        Images are numbered after downsampling, NOT in the original number so they simply go from 0.jpg
        to {chunk_size-1}.jpg.

        eg:

        root/has_anomaly/anomaly_2019-06-11_15-12-00_chunk0/0.jpg
        root/has_anomaly/anomaly_2019-06-11_15-12-00_chunk0/1.jpg
        root/has_anomaly/anomaly_2019-06-11_15-12-00_chunk0/2.jpg
        ...
        """
        for i, chunk in enumerate(self.frames):
            label = chunk['label']
            frames = chunk['frames']

            i_str = str(i).zfill(4)

            write_dir = Path(root) / f"{label}" / f"{self.video_name}_chunk{i_str}"
            os.makedirs(write_dir, exist_ok=True)
            for img, j in frames:
                j_str = str(j).zfill(5)
                img.save(write_dir / f"{j_str}.png","PNG")
        return True



ROOT = "/Users/ludovica/Documents/Insight/data/source_data/"
with open(ROOT+"anomalies.json", 'r') as fin:
    labels_json = json.load(fin)

# Make filename the key, all anomalies are the frame id.
labels = {
    f['filename']: [t['frame_count']
    for t in f['anomalies']] for f in labels_json
}

test_videos = list(Path(ROOT).glob("test/*.avi"))

for p in tqdm.tqdm(test_videos):
    vname = p.name
    vidcap = cv2.VideoCapture(str(p))
    processed = VideoProcessPipeline(
        vidcap,
        vname,
        labels[vname],
        save_every=3,
        scale_to_width=640
    )
    processed.label_frames()
    processed.write_labeled_frames("/Users/ludovica/Documents/Insight/data/my_processed_data_640_slice/test/")


validation_videos = list(Path(ROOT).glob("validation/*.mp4"))
for p in tqdm.tqdm(validation_videos):
    vname = p.name
    vidcap = cv2.VideoCapture(str(p))
    processed = VideoProcessPipeline(
        vidcap,
        vname,
        labels[vname],
        save_every=3,
        scale_to_width=640
    )
    processed.label_frames()
    processed.write_labeled_frames("/Users/ludovica/Documents/Insight/data/my_processed_data_640_slice/validation")
