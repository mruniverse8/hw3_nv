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
from gtts import gTTS
import tempfile
import io

def generate_audio(text, sample_rate=22050):
    try:
        
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_path = fp.name
            tts.save(temp_path)
        waveform, sr = torchaudio.load(temp_path)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        os.remove(temp_path)
        
        return waveform, sample_rate
        
    except ImportError:

        print(f"Warning: gTTS not available. Generating placeholder audio for text: {text[:50]}...")
        duration = len(text.split()) * 0.3
        num_samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, num_samples)
        waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0) 
        return waveform, sample_rate

class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir=None, url_link=None, single_txt=None, *args, **kwargs):
        # assert part == "train_all"
        part = "train_all"
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "custom_dataset"
            data_dir.mkdir(exist_ok=True, parents=True)
        else:
            data_dir = Path(data_dir)
        self._data_dir = data_dir
        self._url_link = url_link
        self._single_txt = single_txt
        
        # If single_txt is provided, create the transcriptions directory with singletxt.txt
        if self._single_txt:
            transcriptions_dir = self._data_dir / "transcriptions"
            transcriptions_dir.mkdir(exist_ok=True, parents=True)
            singletxt_path = transcriptions_dir / "singletxt.txt"
            with singletxt_path.open("w") as f:
                f.write(self._single_txt)
            singletxt_path = transcriptions_dir / "singletxt1.txt" #we need batch of size 2:P
            
            with singletxt_path.open("w") as f:
                f.write(self._single_txt)
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

            if not wav_path.exists():
                waveform, sample_rate = generate_audio(text)
                torchaudio.save(str(wav_path), waveform, sample_rate)
            
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