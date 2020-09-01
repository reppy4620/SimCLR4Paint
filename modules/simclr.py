import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)


class SimCLR(nn.Module):

    def __init__(self, projection_dim):
        super(SimCLR, self).__init__()

        model = torch.hub.load('RF5/danbooru-pretrained', 'resnet18')
        # model has 2 nn.Sequential()
        # first part is resnet body
        # second part is fc module added by fastai
        # so i need only first part, resnet body.
        layers = [l for l in model.children()]
        model = nn.Sequential(
            layers[0],
            nn.AdaptiveAvgPool2d(1),  # equal to Global Average Pooling
            Flatten(),  # Flatten module like Keras
            nn.Linear(512, projection_dim, bias=False)
        )
        self.encoder = model

        self.projector = nn.Sequential(
            nn.Linear(projection_dim, projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim, bias=False),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z
