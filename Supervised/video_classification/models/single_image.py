from torch import nn
from torchvision.models import vgg16
from .mlp import MLP


class SingleImageModel(nn.Module):
    def __init__(self, mlp_sizes=[768, 128, 2]):
        super().__init__()
        self.vgg = vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(
            self.vgg.classifier[:-1])  # Remove imagenet output layer
        in_features = 4096  # vgg feats
        self.clf = MLP(in_features, mlp_sizes)
        self.freeze_vgg()

    def forward(self, x):
        x = self.vgg(x)
        x = self.clf(x)
        return x

    def freeze_vgg(self):
        # Freeze the VGG classifier
        for p in self.vgg.parameters():
            p.requires_grad = False

    def unfreeze_vgg(self):
        # Unfreeze the VGG classifier.
        # Training the whole VGG is a no-go, so
        # we only train the classifier part.
        for p in self.vgg.classifier[1:].parameters():
            p.requires_grad = True
