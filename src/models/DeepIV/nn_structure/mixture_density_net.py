import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, OneHotCategorical

import logging

logger = logging.getLogger()

class MixtureDensityNet(nn.Module):

    def __init__(self, n_input: int, n_output: int, n_component: int):
        """

        Parameters
        ----------
        n_input : int
            the dimension of input feature
        n_output :
            the dimension of output space
        n_component :
            the number of component of Gauss distribution
        """
        super(MixtureDensityNet, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_component = n_component
        self.mu_linear = nn.Linear(n_input, n_output * n_component)
        self.logsigma_linear = nn.Linear(n_input, n_output * n_component)
        self.logpi_linear = nn.Linear(n_input, n_component)

    def forward(self, feature):
        """

        Parameters
        ----------
        feature : torch.Tensor[N, n_input]
            input feature

        Returns
        -------
        norm: Normal
        cat: OneHotCategorical
        """
        n_data = feature.size()[0]
        mu = self.mu_linear(feature).reshape((n_data, self.n_output, self.n_component))
        logsigma = self.logsigma_linear(feature).reshape((n_data, self.n_output, self.n_component))
        norm = Normal(loc=mu, scale=torch.exp(logsigma))
        logpi = self.logpi_linear(feature)
        logpi = logpi - torch.min(logpi)
        cat = OneHotCategorical(logits=logpi)
        return norm, cat
