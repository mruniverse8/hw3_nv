import json
import os
import shutil
from pathlib import Path

import torch
import torchaudio
import wget
import yadisk
from tqdm import tqdm
import zipfile

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
import nemo.collections.tts as nemo_tts

_tts_model = None
_vocoder_model = None

def generate_audio(text, sample_rate=22050):
    """
    Generate audio from text using NVIDIA NeMo acoustic models.
    Uses FastPitch for mel-spectrogram generation and HiFiGAN for vocoding.
    """
    global _tts_model, _vocoder_model
    
    if _tts_model is None:
        print("Loading NVIDIA NeMo TTS models...")
        _tts_model = nemo_tts.models.FastPitchModel.from_pretrained("tts_en_fastpitch")
        _vocoder_model = nemo_tts.models.HifiGanModel.from_pretrained("tts_en_hifigan")
        _tts_model.eval()
        _vocoder_model.eval()
    
    parsed = _tts_model.parse(text)
    spectrogram = _tts_model.generate_spectrogram(tokens=parsed)
    audio = _vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)
    
    waveform = audio.cpu()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    model_sample_rate = _vocoder_model.cfg.sample_rate
    if model_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(model_sample_rate, sample_rate)
        waveform = resampler(waveform)
    
    return waveform, sample_rate

class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir=None, url_link=None, *args, **kwargs):
        part = "newdata"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "custom_dataset"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._url_link = url_link
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        if self._url_link:
            arch_path = self._data_dir / f"{part}.zip"
            try:
                y = yadisk.YaDisk()
                y.download(self._url_link, str(arch_path))
            except Exception:
                try:
                    wget.download(self._url_link, str(arch_path))
                except Exception as e:
                    raise ValueError(f"Failed to download from {self._url_link}: {e}")
            
            extract_dir = self._data_dir / "temp_extract"
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(arch_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            gt_audio_dirs = list(extract_dir.rglob("gt_audio"))
            transcriptions_dirs = list(extract_dir.rglob("transcriptions"))
            
            if not gt_audio_dirs or not transcriptions_dirs:
                raise ValueError("Extracted archive does not contain 'gt_audio' and 'transcriptions' directories.")
            
            src_gt_audio = gt_audio_dirs[0]
            src_transcriptions = transcriptions_dirs[0]
            
            target_gt_audio = self._data_dir / "gt_audio"
            target_transcriptions = self._data_dir / "transcriptions"
            
            if target_gt_audio.exists():
                shutil.rmtree(target_gt_audio)
            if target_transcriptions.exists():
                shutil.rmtree(target_transcriptions)
                
            shutil.move(str(src_gt_audio), str(target_gt_audio))
            shutil.move(str(src_transcriptions), str(target_transcriptions))
            
            # Clean up
            shutil.rmtree(extract_dir)
            arch_path.unlink()

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        dataset_dir = self._data_dir
        gt_audio_dir = dataset_dir / "gt_audio"
        transcriptions_dir = dataset_dir / "transcriptions"
        
        if not transcriptions_dir.exists():
            self._load_part(part)
        
        gt_audio_dir.mkdir(exist_ok=True)
        
        for txt_path in transcriptions_dir.glob("*.txt"):
            base_name = txt_path.stem
            
            with txt_path.open() as f:
                text = f.read().strip().lower()
            
            wav_path = gt_audio_dir / f"{base_name}.wav"
            
            # Generate audio if it doesn't exist
            if not wav_path.exists():
                waveform, sample_rate = generate_audio(text)
                torchaudio.save(str(wav_path), waveform, sample_rate)
            
            # Add to index if wav file exists
            if wav_path.exists():
                t_info = torchaudio.info(str(wav_path))
                length = t_info.num_frames / t_info.sample_rate
                index.append(
                    {
                        "path": str(wav_path),
                        "text": text,
                        "audio_len": length,
                    }
                )
        
        return index