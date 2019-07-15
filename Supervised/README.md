# SupervisedVideoClassification
This project uses PyTorch to classify short videos in a supervised fashion.

The core of this approach is the following: we load a sliding window of N images,
and we have models that take those N images and produce a prediction on the
*last* image in the window (so that we predict timestep t with context from the past, without seeing the future!).

The first step is to get a video (eg in .mp4) and decode it into a format called **FrameFolder**.

The format is the following: each video's images go in a folder and all metadata (eg labels) go into a `data.csv` file inside that folder. Images go into a 'images' subfolder eg:

`video_a.mp4` would go to:

`video_a/`
Labels would be in `video_a/data.csv`
All frames are saved as PNGs inside `video_a/images/`.


This is supposed to be done only once! The code to do that is in `utils/videos_to_frame_folders.py`. If you have 3 videos, you are going to have 3 folders, each with all the images for that video. To populate `data.csv`, that code reads labels from a JSON file that has the start and end frames for each anomaly. From these, we derive what frames are positive and what frames are negative.

After that, we simply consume data as a PyTorch Dataset. We have two types:

a. `FrameFolder`. Every `__getitem__` returns a single frame with its label.
b. `FrameWindow`. Every `__getitem__` returns a window as a 4D tensor.

Since we have many videos, we want to read all of them into a single large Dataset.
To do that, all we have to do is read and concatenate and we do that in the `FolderOfFrameFolders`.
You can pass which of the base classes you want to have.

Each experiment has its own notebook with all the logic (and results!) in there.
