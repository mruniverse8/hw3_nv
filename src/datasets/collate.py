import torch


import torch.nn.utils.rnn as rnn_utils

import torch
import torch.nn.utils.rnn as rnn_utils

def collate_fn(dataset_items: list[dict]):
    pad_value = -11.5129251  
    spectrograms = [
        item["spectrogram"].squeeze(0).transpose(0, 1) 
        for item in dataset_items
    ]
    spectrograms_padded = rnn_utils.pad_sequence(
        spectrograms, 
        batch_first=True, 
        padding_value=pad_value
    )
    spectrograms_padded = spectrograms_padded.permute(0, 2, 1)
    audios = [item["audio"].squeeze(0) for item in dataset_items]
    audios_padded = rnn_utils.pad_sequence(
        audios, 
        batch_first=True, 
        padding_value=0.0
    ).squeeze(0)#mono

    return {
        "spectrogram": spectrograms_padded,
        "audio": audios_padded,            
        "audio_path": [item["audio_path"] for item in dataset_items],
    }