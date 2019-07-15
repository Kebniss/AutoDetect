import torch
from torch import nn
from .single_image import SingleImageModel
from .mlp import MLP


class MultiImageModel(nn.Module):
    def __init__(self,
                 window_size=3,
                 single_mlp_sizes=[768, 128],
                 joint_mlp_sizes=[64, 2]):
        super().__init__()
        self.window_size = window_size
        self.single_mlp_sizes = single_mlp_sizes
        self.joint_mlp_sizes = joint_mlp_sizes

        self.single_image_model = SingleImageModel(self.single_mlp_sizes)
        self.in_features = self.single_mlp_sizes[-1] * self.window_size
        self.clf = MLP(self.in_features, joint_mlp_sizes)

    def forward(self, x):
        # x is of size [B, T, C, H, W]. In other words, a batch of windows.
        # each img for the same window goes through SingleImageModel
        x = x.transpose(0, 1)  # -> [T, B, C, H, W]
        x = torch.cat([self.single_image_model(window) for window in x], 1)
        # x is now of size [B, T * single_mlp_sizes[-1]]

        x = self.clf(x)
        # Now size is [B, joint_mlp_sizes[-1]] which should always be 2

        return x
