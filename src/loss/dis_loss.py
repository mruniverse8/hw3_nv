import torch
from torch import Tensor
import torch.nn.functional as F
from src.transforms.wav_augs.mel import MelSpectrogram

class DisLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(
        self, generator, discriminators, audio, spectrogram, pred_wav, **batch
    ) -> Tensor:
    
    loss = None
    with torch.no_grad():
            pred_detached = pred_wav.detach()
    
    for discriminator in discriminators:
    
        D_G_s, _ = discriminator(pred_detached, feature_extraction = False)
        D_x, _ = discriminator(audio, feature_extraction = False)
    
        part_loss = torch.mean((D_x - 1) ** 2 + D_G_s ** 2) #GAN loss
    
        if loss is None:
            loss = part_loss
        else:
            loss += part_loss
    
    assert loss is not None

    return {"dis_loss": loss}
