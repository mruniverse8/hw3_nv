import torch
from torch import Tensor
import torch.nn.functional as F
from src.transforms.wav_augs.mel import MelSpectrogram

class GANLoss(nn.Module):
    def __init__(self, lambda_ft=2, lambda_mel=45):
        super().__init__()
        self.lambda_ft = lambda_ft
        self.lambda_mel = lambda_mel
    def forward(
        self, generator, discriminators, audio, spectrogram, pred_wav, **batch
    ) -> Tensor:
    
    #GANLOSS
    
    mel_orig = MelSpectrogram(audio)
    mel_pred = MelSpectrogram(pred_wav)
    loss_mel = lambda_mel * F.l1_loss(mel_pred, mel_orig) #MEL-spectrogram loss
    for discriminator in discriminators:
        D_G_s, F_G_s_i = discriminator(pred_wav,feature_extraction = True)
        D_x, F_x_i = discriminator(audio,feature_extraction = True)
        assert(len(F_G_s_i) == len(F_x_i))
        for feat_x, feat_pred in zip(F_x_i, F_G_s_i):
            loss += lambda_ft * F.l1_loss(feat_pred, feat_x)
        loss += torch.mean((D_G_s - 1) **2) #GAN loss

    return {"gan_loss": loss}
