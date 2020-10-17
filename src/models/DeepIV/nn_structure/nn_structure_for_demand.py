import torch
from torch import nn

from ..nn_structure.mixture_density_net import MixtureDensityNet


class ResponseModel(nn.Module):

    def __init__(self, dropout_ratio):
        super(ResponseModel, self).__init__()
        self.net = nn.Sequential(nn.Linear(3, 128),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_ratio),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_ratio),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_ratio),
                                 nn.Linear(32, 1))

    def forward(self, treatment, covariate):
        feature = torch.cat([treatment, covariate], dim=1)
        return self.net(feature)


def build_net_for_demand(dropout_rate, **args):
    instrumental_net = nn.Sequential(nn.Linear(3, 128),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_rate),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_rate),
                                     nn.Linear(64, 32),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_rate),
                                     MixtureDensityNet(32, 1, 10))

    response_net = ResponseModel(dropout_rate)
    return instrumental_net, response_net
