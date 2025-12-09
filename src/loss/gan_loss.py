import torch
from torch import Tensor
import torch.nn.functional as F
from src.transforms.wav_augs.mel import MelSpectrogram
from src.transforms.wav_augs.mel import MelSpectrogramConfig

def pad_to_same_length(tensor1: Tensor, tensor2: Tensor, pad_value=-11.5129251) -> tuple[Tensor, Tensor]:

    len1 = tensor1.shape[-1]
    len2 = tensor2.shape[-1]
    max_len = max(len1, len2)
    if len1 < max_len:
        pad_size = max_len - len1
        tensor1 = F.pad(tensor1, (0, pad_size), value=pad_value)
    if len2 < max_len:
        pad_size = max_len - len2
        tensor2 = F.pad(tensor2, (0, pad_size), value=pad_value)
    return tensor1, tensor2

class GANLoss(torch.nn.Module):
    def __init__(self, lambda_ft=2, lambda_mel=45):
        super().__init__()
        self.lambda_ft = lambda_ft
        self.lambda_mel = lambda_mel
        self.mel_spec = MelSpectrogram(MelSpectrogramConfig())

    def forward(
        self, generator, discriminators, audio, spectrogram, pred_wav, **batch
    ) -> Tensor:

        lambda_mel = self.lambda_mel
        lambda_ft = self.lambda_ft
        audio = audio.squeeze(0)
        ##print(audio.shape)
        ##print(pred_wav.shape)
        mel_orig = self.mel_spec(audio)
        mel_pred = self.mel_spec(pred_wav)
        ##print(mel_orig.shape)
        ##print(mel_pred.shape)
        mel_orig, mel_pred = pad_to_same_length(mel_orig, mel_pred)
        pred_wav, audio = pad_to_same_length(pred_wav, audio, 0)
        ##print(mel_orig.shape)
        ##print(mel_pred.shape)
        ##print(audio.shape)
        ##print(pred_wav.shape)
        loss  = lambda_mel * F.l1_loss(mel_pred, mel_orig) #MEL-spectrogram loss
        ##print("still in gan loss")
        for discriminator in discriminators:
            D_G_s, F_G_s_i = discriminator(pred_wav,feature_extraction = True)
            D_x, F_x_i = discriminator(audio,feature_extraction = True)
            assert(len(F_G_s_i) == len(F_x_i))
            for feat_x, feat_pred in zip(F_x_i, F_G_s_i):
                loss += lambda_ft * F.l1_loss(feat_pred, feat_x)
            loss += torch.mean((D_G_s - 1) **2) #GAN loss
        ##print(f"losss {loss} end gan")
        return {"gan_loss": loss}
