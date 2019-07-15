from torch import nn
from .single_image import SingleImageModel


class AverageImagesModel(nn.Module):
    def __init__(self, mlp_sizes=[768, 128, 2]):
        super().__init__()
        self.single_image_model = SingleImageModel(mlp_sizes)

    def forward(self, x):
        # x is of size (B, T, C, H, W)
        x = x.mean(1)  # We average all images in axis T
        x = self.single_image_model(x)  # and then it's business as usual
        return x

    def freeze_vgg(self):
        # Freeze the VGG classifier
        self.single_image_model.freeze_vgg()

    def unfreeze_vgg(self):
        self.single_image_model.unfreeze_vgg()
