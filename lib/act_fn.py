import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def tanlu(x, cap=1):
    return torch.maximum(0, torch.minimum(x, (cap-1)+torch.tanh(x-(cap-1))))