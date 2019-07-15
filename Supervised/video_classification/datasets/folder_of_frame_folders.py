from torch.utils.data import ConcatDataset
from pathlib import Path
from .frame_folder import FrameFolderDataset


class FolderOfFrameFoldersDataset(ConcatDataset):
    """
    If you put many FrameFolderDatasets in the same folder, we can read
    all of them using this class as if they were a single large dataset.
    """
    def __init__(self,
                 root,
                 label_itos=['negative', 'positive'],
                 transform=None,
                 base_class=FrameFolderDataset,
                 **base_class_kwargs,
                 ):
        self.root = Path(root)
        self.label_itos = label_itos
        self.transform = transform

        videos_paths = [
            data_csv.parent for data_csv in self.root.glob("*/data.csv")
        ]
        datasets = [
            base_class(
                p, label_itos=label_itos, transform=transform, **base_class_kwargs)
            for p in videos_paths
        ]
        super().__init__(datasets)

    def __repr__(self):
        freqs = [d.class_frequencies for d in self.datasets]
        distribution = {
            label_class: sum(item.get(label_class, 0) for item in freqs)
            for label_class in self.label_itos
        }
        message = (
            f"FolderOfFrameFoldersDataset with {len(self)} samples.\n"
            f"\tOverall data distribution: {distribution}")
        return message
