from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, mlp_sizes=[768, 128, 2], dropout=0.3):
        super().__init__()
        in_features = input_size  # vgg feats
        out_features = mlp_sizes[0]

        layers = []
        for i, size in enumerate(mlp_sizes):
            out_features = mlp_sizes[i]

            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_features)),
            layers.append(nn.Dropout(p=dropout))
            in_features = out_features

        layers.pop()  # Remove last dropout
        layers.pop()  # Remove last BN
        layers.pop()  # Remove last ReLU
        self.clf = nn.Sequential(*layers)

    def forward(self, x):
        x = self.clf(x)
        return x
