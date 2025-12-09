import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class DiscriminatorMSD(nn.Module):
    def __init__(self, class_normalization,pool_factor=None):
        super().__init__()
        if class_normalization is None:
            class_normalization = weight_norm
        self.idx_features = []
        self.pool_factor = pool_factor
        #do we need to pad? I think no
        self.layers = nn.Sequential(
            class_normalization(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),
            nn.LeakyReLU(),

            class_normalization(nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20)),
            nn.LeakyReLU(),

            class_normalization(nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20)),
            nn.LeakyReLU(),

            class_normalization(nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=20)),
            nn.LeakyReLU(),

            class_normalization(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20)),
            nn.LeakyReLU(),

            class_normalization(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(),

            class_normalization(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))
        )
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                self.idx_features.append(len(self.layers))

    def forward(self, x, feature_extraction=False):
        x = x.unsqueeze(1) #[B,1,T]?
        if self.pool_factor is not None:
            x = nn.functional.avg_pool1d(x, self.pool_factor)
        features = []
        for i, layer in enumerate(self.layers):
            if  feature_extraction==True and i in self.idx_features:
                features.append(x)
            x = layer(x)
        x = torch.mean(x, dim=2) # should be done in this way? or shape will match?
        # [B,1,1] is it correct?
        return x, features