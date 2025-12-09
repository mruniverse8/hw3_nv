import json
import os
import shutil
from pathlib import Path

import torchaudio
import wget
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "train_all": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
}


class LJSpeechDataset(BaseDataset):
    def __init__(self, part="train_all", data_dir=None, *args, **kwargs):
        assert part == "train_all"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.bz2"
        print(f"Loading part {part}")
        if not arch_path.exists():
            wget.download(URL_LINKS[part], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir).iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        #os.remove(str(arch_path))
        #shutil.rmtree(str(self._data_dir))

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
        if not (dataset_dir / "metadata.csv").exists():
            self._load_part(part)

        metadata_path = dataset_dir / "metadata.csv"
        wavs_dir = dataset_dir / "wavs"
        with metadata_path.open() as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 3:
                    id_, trans, norm_trans = parts # trans needed?
                    wav_path = wavs_dir / f"{id_}.wav"
                    if wav_path.exists():
                        t_info = torchaudio.info(str(wav_path))
                        length = t_info.num_frames / t_info.sample_rate
                        index.append(
                            {
                                "path": str(wav_path),
                                "text": norm_trans.lower(),
                                "audio_len": length, 
                            }
                        )
        return index
