import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class DiscriminatorMPD(nn.Module):
    """Multi-Period Discriminator with weight normalization for training stability."""
    
    def __init__(self, p_fact):
        super().__init__()
        self._p = p_fact
        self.layers = nn.ModuleList()
        self.idx_features = []  # index of conv layers for feature extraction
        h_prev = 1
        for i in range(1, 5):
            self.idx_features.append(len(self.layers))
            # Apply weight normalization to all Conv2d layers
            self.layers.append(weight_norm(nn.Conv2d(h_prev, 2 ** (5 + i), (5, 1), stride=(3, 1), padding=(2, 0))))
            self.layers.append(nn.LeakyReLU(0.1))
            h_prev = 2 ** (5 + i)
        self.layers.append(weight_norm(nn.Conv2d(h_prev, 1024, (5, 1))))
        self.idx_features.append(len(self.layers))
        self.layers.append(nn.LeakyReLU(0.1))
        self.layers.append(weight_norm(nn.Conv2d(1024, 1, (3, 1))))
        self.idx_features.append(len(self.layers))

    def reshape_with_padding(self, x):
        p = self._p
        B, C, T = x.shape
        assert C == 1, "shape error reshape_pad"
        pad_len = (p - (T % p)) % p
        if pad_len:
            x = torch.nn.functional.pad(x, (0, pad_len))
        return x.view(B, C, -1, p)

    def forward(self, x, feature_extraction=False):
        features = [] #can be torch?
        x = x.unsqueeze(1)
        x = self.reshape_with_padding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if feature_extraction and i in self.idx_features:
                features.append(x)
        # The vector is shape (B,1, H,W)?
        x = x.view(x.size(0), -1)
        # The vector comparison should it be done via? mean average? or vector of features?
        # less multiplication of gradients  will mean less gradient explotion?
        return x, features