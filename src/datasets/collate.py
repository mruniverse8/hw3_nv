import torch


import torch.nn.utils.rnn as rnn_utils

def collate_fn(dataset_items: list[dict]):
    """
    Assumes:
        - audio: [1, T_audio]
        - spectrogram: [n_mels, T_spec]
    """
    audios = [item["audio"].squeeze(0) for item in dataset_items]
    spectrograms = [item["spectrogram"] for item in dataset_items]
    texts = [item["text"] for item in dataset_items]
    paths = [item["audio_path"] for item in dataset_items]
    
    pad_value = -11.5129251 # Taken from config best way?
    spectrograms_padded = rnn_utils.pad_sequence(
        spectrograms, batch_first=True, padding_value=pad_value
    )

    audios_padded = rnn_utils.pad_sequence(
        audios, batch_first=True, padding_value=0.0
    ).unsqueeze(1)

    # spec_lengths = torch.tensor([s.size(-1) for s in spectrograms])
    # audio_lengths = torch.tensor([a.size(-1) for a in audios])

    return {
        "spectrogram": spectrograms_padded,
        "audio": audios_padded,
        #"spectrogram_lengths": spec_lengths,
        #"audio_lengths": audio_lengths,
        #"text": texts,
        "audio_path": paths,
    }