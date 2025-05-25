
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    def init_weights(self):
        pass


@abstractmethod
def forward(self, x):
    """Forward function.

    Args:
        x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
            torch.Tensor, containing input data for forward computation.
    """
