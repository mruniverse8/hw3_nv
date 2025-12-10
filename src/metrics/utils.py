# Based on seminar materials

# Don't forget to support cases when target_text == ''
import torchaudio
import numpy
from matplotlib import pyplot as plt

def _save_spectrogram_image(spectrogram, save_path, title="Spectrogram"):
    """
    Plot and save a spectrogram as an image.
    
    Args:
        spectrogram (torch.Tensor): Spectrogram tensor of shape [n_mels, T]
        save_path (str or Path): Path to save the image
        title (str): Title for the plot
    """
    plt.figure(figsize=(12, 4))
    spec_np = spectrogram.cpu().numpy()
    plt.imshow(spec_np, aspect='auto', origin='lower', interpolation='nearest')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency Bin')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
def _save_audio( audio, save_path, sample_rate=22050):
    """
    Save audio tensor to a wav file.
    
    Args:
        audio (torch.Tensor): Audio tensor of shape [T] or [1, T]
        save_path (str or Path): Path to save the audio file
        sample_rate (int): Sample rate of the audio
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio.cpu()
    torchaudio.save(str(save_path), audio, sample_rate)

def calc_cer(target_text, predicted_text) -> float:
    # TODO
    pass


def calc_wer(target_text, predicted_text) -> float:
    # TODO
    pass

