from torch import nn
from torchvision.models import resnet101
from .mlp import MLP


class SingleImageResNetModel(nn.Module):
    def __init__(self, mlp_sizes=[768, 128, 2]):
        super().__init__()
        resnet = resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.clf = MLP(2048, mlp_sizes)
        self.freeze_resnet()

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.clf(x)
        return x

    def freeze_resnet(self):
        for p in self.resnet.parameters():
            p.requires_grad = False

    def unfreeze_resnet(self):
        for p in self.resnet.parameters():
            p.requires_grad = True
