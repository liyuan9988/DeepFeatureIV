import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .mixture_density_net import MixtureDensityNet


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class ImageFeature(nn.Module):

    def __init__(self):
        super(ImageFeature, self).__init__()
        self.treatment_net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32 * 4 * 4)),  # B, 512
            nn.BatchNorm1d(32 * 4 * 4),
            nn.Linear(32 * 4 * 4, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 128),  # B, 256
            nn.ReLU(True),
            nn.Linear(128, 32),  # B, z_dim*2
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

    def forward(self, data):
        image = data.reshape((-1, 1, 64, 64))
        return self.treatment_net(image)


def build_net_for_dsprite():
    response_net = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                 nn.ReLU(),
                                 spectral_norm(nn.Linear(1024, 512)),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(512),
                                 spectral_norm(nn.Linear(512, 128)),
                                 nn.ReLU(),
                                 spectral_norm(nn.Linear(128, 32)),
                                 nn.BatchNorm1d(32),
                                 nn.Tanh())

    # treatment_net = ImageFeature()
    instrumental_net = nn.Sequential(spectral_norm(nn.Linear(3, 256)),
                                     nn.ReLU(),
                                     spectral_norm(nn.Linear(256, 128)),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(128),
                                     spectral_norm(nn.Linear(128, 128)),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(128),
                                     spectral_norm(nn.Linear(128, 32)),
                                     nn.BatchNorm1d(32),
                                     nn.ReLU(),
                                     MixtureDensityNet(32, 64 * 64, 10))

    covariate_net = None

    return instrumental_net, response_net
