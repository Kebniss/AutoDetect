import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader as read_image
from pathlib import Path
import torch


class FrameWindowDataset(Dataset):
    """
    Just like FrameFolderDataset, but its output is different.

    Here, for every frame t, we return a 4D tensor of size [T, C, H, W] where:
    T is the "time" component (ie multiple frames)
    C, H, W are the usual dimensions for an image (color, height, width).

    The size of T is determined by the window size.
    Given we can't see the future, we stack every frame at time t
    with t-1, t-2, ..., t-window_size and align it with the label for t.

    Windows do overlap each other.
    """

    def __init__(
            self,
            root,
            label_itos=['negative', 'positive'],
            transform=None,
            window_size=3,
            overlapping=True,
    ):
        self.root = Path(root)
        self.label_itos = label_itos
        self.label_stoi = {label: i for i, label in enumerate(self.label_itos)}

        self.transform = transform

        self.window_size = window_size
        self.overlapping = overlapping

        self.chunks = self._chunkify()

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]

        label = self.label_stoi[chunk.iloc[-1][
            'label']]  # Cannot see future, so label comes from last from input

        images = [
            read_image(self.root / image_path)
            for image_path in chunk['image_path']
        ]

        if self.transform:
            images = [self.transform(image) for image in images]
        return torch.stack(images, dim=0), label

    def _chunkify(self):
        df = pd.read_csv(self.root / 'data.csv')
        subsets = []
        offset = 1 if self.overlapping else self.window_size
        for start in range(0, df.shape[0], offset):
            if df.shape[0]-start < self.window_size:
                break
            df_subset = df.iloc[start:start + self.window_size]
            subsets.append(df_subset)
        return subsets

    def __repr__(self):
        message = (f"FrameWindowDataset with {len(self)} samples.\n")
        return message
