import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader as read_image
from pathlib import Path


class FrameFolderDataset(Dataset):
    """
    Reads all frame from a single video assuming the FrameFolder format.
    To learn more, see utils/videos_to_frame_folders.py which is the code
    that turns a folder of video files (eg .mp4) into a folder of FrameFolders.
    """
    def __init__(self,
                 root,
                 label_itos=['negative', 'positive'],
                 transform=None):
        self.root = Path(root)
        self.df = pd.read_csv(self.root / 'data.csv')
        self.label_itos = label_itos
        self.label_stoi = {label: i for i, label in enumerate(self.label_itos)}
        self.class_frequencies = self.df['label'].value_counts().to_dict()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.root / row['image_path']
        image = read_image(image_path)
        label = self.label_stoi[row['label']]

        if self.transform:
            image = self.transform(image)
        return image, label

    def __repr__(self):
        message =  (
            f"FrameFolderDataset with {len(self)} samples.\n"
            f"\tData distribution: {self.class_frequencies}"
        )
        return message
