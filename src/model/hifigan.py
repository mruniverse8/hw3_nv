from typing import List
import torch
from torch import nn
from torch.nn import Sequential
from torch.nn.utils import weight_norm


class MRFBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k_r, D_r: List):
        super().__init__()
        self.lst_block = nn.ModuleList()
        for m in range(len(D_r)):
            for l in range(len(D_r[m])):
                d_aux = D_r[m][l]
                assert (d_aux * (k_r - 1)) % 2 == 0  # cond to fix padding independent of kernel and dilation

                auxlayer = nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(nn.Conv1d(ch_in, ch_out, kernel_size=k_r, #todo check if improves
                                         dilation=D_r[m][l], padding=(d_aux*(k_r-1))//2))
                )
                self.lst_block.append(auxlayer)
                
    def forward(self, x: torch.Tensor):
        for block in self.lst_block:
            x = x + block(x)  # Skip connection
        return x


class ResBlock(nn.Module):
    """Residual block with weight normalization for training stability."""

    def __init__(self, h_u_prev, h_u_next, k_u, k_r, D_r):
        super().__init__()
        self.h_u_prev = h_u_prev
        self.h_u_next = h_u_next
        self.k_u = k_u
        self.k_r = k_r
        self.D_r = D_r
        self.lrel = nn.LeakyReLU(0.1)  # Slope 0.1 for better gradient flow
        assert k_u % 2 == 0
        self.conv1 = weight_norm(nn.ConvTranspose1d(h_u_prev, h_u_next, kernel_size=k_u, stride=(k_u//2)))
        assert len(k_r) == len(D_r)
        self.mrf_blocks = nn.ModuleList()
        for i in range(len(k_r)):
            self.mrf_blocks.append(MRFBlock(h_u_next, h_u_next, k_r[i], D_r[i]))
    
    def forward(self, x: torch.Tensor):
        x = self.lrel(x)
        x = self.conv1(x)
        y = None
        for mrf in self.mrf_blocks:
            if y is None:
                y = mrf(x)
            else:
                y = y + mrf(x)
        y = y / len(self.mrf_blocks) 
        return y


class HIFIGAN(nn.Module):
    """
    HiFi-GAN Generator with weight normalization for training stability.
    """

    def __init__(self, sample_rate, n_mels, h_u, k_u, k_r, D_r):
        """
        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel spectrogram bins
            h_u: Initial hidden dimension
            k_u: List of upsampling kernel sizes
            k_r: List of residual kernel sizes
            D_r: List of dilation rates for residual blocks
        """
        super().__init__()
        assert len(k_r) == len(D_r)
        self._sample_r = sample_rate
        self._n_mels = n_mels
        self.pre_conv = weight_norm(nn.Conv1d(n_mels, h_u, kernel_size=7, stride=1, padding=3))

        self.res_block = nn.ModuleList()
        h_u_prev = h_u
        for i in range(len(k_u)):
            self.res_block.append(ResBlock(h_u_prev, h_u_prev // 2, k_u[i], k_r, D_r))
            h_u_prev = h_u_prev // 2
        self.lrel = nn.LeakyReLU(0.1)
        self.post_conv = weight_norm(nn.Conv1d(h_u_prev, 1, kernel_size=7, stride=1, padding=3))
        self.tanh = nn.Tanh()

        self.apply(self._init_weights)



    def forward(self, x: torch.Tensor):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        x = self.pre_conv(x)
        for i in range(len(self.res_block)):
            x = self.res_block[i](x)
        x = self.lrel(x)
        x = self.post_conv(x)
        x = self.tanh(x)
        x_trim = x.squeeze(1)
        #assert x_trim.shape == (x.shape[0], self._sample_r), f"Error shape trim x {x.shape}"
        return {"pred_wav": x_trim}

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
