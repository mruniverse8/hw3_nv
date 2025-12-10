# Demo Explanation for syntesize.py

This document provides a step-by-step guide to setting up and using the `syntesize.py` script for text-to-speech synthesis.
## 0 Git checkout
```bash
git checkout synthesizer
```

## 1. Install Requirements

First, install the necessary Python packages by running the following command in your terminal:

```bash
pip install -r requirements.txt
pip install -r requirements2.txt
```

This will install all dependencies listed in `requirements.txt`, including PyTorch, torchaudio, hydra-core, and other libraries required for the project.


## 2. Download Weights

The model weights need to be downloaded to perform synthesis. Following the example template in `trainer.ipynb`, use the `download_weights.py` script to download and extract the weights.

Assuming the notebook provides a URL for the weights (e.g., a ZIP file containing the checkpoint), run:

```bash
python download_weights.py <weights_download_url> -o saved_more
```

Replace `<weights_download_url>` with the actual URL specified in the notebook. This will download the ZIP file and extract it to the `saved_more` directory.

If the `saved_more` directory already exists with the required checkpoints (e.g., `checkpoint-epoch3.pth`), you can skip this step.

```bash
!python download_weights.py "https://drive.usercontent.google.com/download?id=1KIp2IhbuiEU68g83pJ7G40FoUMFnIMlX&export=download&authuser=0&confirm=t&uuid=066f80fc-feb3-4dc1-8420-8d35dd70fc64&at=ALWLOp7pkhBvjTUJtTJZ7MrvmJ54:1765400324028" -o .
```



## 3. Run Synthesis

The `syntesize.py` script uses Hydra for configuration management. To perform synthesis on a single text input, run the following command:

```bash
python syntesize.py --config-name sint datasets.train.single_txt="Your text to synthesize here" trainer.from_pretrained="saved_more/testing/checkpoint-epoch5.pth"
```

### Explanation of Arguments:
- `--config-name sint`: Specifies the configuration file `sint.yaml` located in `src/configs/`, which is set up for synthesis using the HiFi-GAN model and the `sintesizer` dataset.
- `datasets.train.single_txt="Your text to synthesize here"`: Sets the text input for synthesis. Replace with your desired text.
- `trainer.from_pretrained="saved_more/testing/checkpoint-epoch5.pth"`: Points to the pretrained model checkpoint. Adjust the path if your checkpoint is in a different location.

### Example:
```bash
python syntesize.py --config-name sint datasets.train.single_txt="Hello, world! This is a test of text-to-speech synthesis." trainer.from_pretrained="saved_more/testing/checkpoint-epoch5.pth"
```

This will run inference, generate the audio, and output details such as the audio shape, sample rate, and duration to the console. The generated audio files will be saved in `inferece_data/saved/saved_more/` (or the configured save_dir).

### Using a Dataset Directory:
If you prefer to synthesize from a dataset directory instead of a single text, you can set the data directory:

```bash
python syntesize.py --config-name sint datasets.train.data_dir="test_data2" trainer.from_pretrained="saved_more/testing/checkpoint-epoch5.pth"
```

This will process all text files in the specified directory (e.g., `test_data2/transcriptions/`) and generate corresponding audio.

### Automatic Dataset Download:
It is also possible to set a download URL for the dataset either by editing `src/configs/datasets/sintesizer.yaml` or via command line arguments.

To edit the YAML file, set `url_link` to the desired URL:

```yaml
train:
  _target_: src.datasets.CustomDirDataset
  data_dir: null
  url_link: "https://example.com/dataset.zip"  # Replace with your dataset download URL
  single_txt: null
  instance_transforms:
    get_spectrogram:
      _target_: src.transforms.wav_augs.MelSpectrogram
      config:
        _target_: src.transforms.wav_augs.MelSpectrogramConfig
```

Alternatively, you can set it via command line: `datasets.train.url_link="your_url"`
```bash
python syntesize.py --config-name sint datasets.train.url_link="your_url" trainer.from_pretrained="saved_more/testing/checkpoint-epoch5.pth"
```

When `url_link` is set, the `CustomDirDataset` will attempt to download the file from the URL (using Yandex Disk or wget as fallback) and extract it to the `data_dir` if specified. This is useful for automatically fetching datasets without manual download steps.

### Reproducing Experiments from the Report:
To reproduce the experiments in the report, run the following commands. These examples use batch size 2, process specific test datasets, and save results to designated output directories:

```bash
HYDRA_FULL_ERROR=1 python3 syntesize.py -cn=sint dataloader.batch_size=2 datasets.train.data_dir=test_data3 trainer.save_dir=dataout3 trainer.from_pretrained=saved_more/testing/checkpoint-epoch4.pth
```

```bash
HYDRA_FULL_ERROR=1 python3 syntesize.py -cn=sint dataloader.batch_size=2 datasets.train.data_dir=test_data2 trainer.save_dir=dataout2 trainer.from_pretrained=saved_more/testing/checkpoint-epoch4.pth
```

These commands synthesize audio from the text files in `test_data3` and `test_data2`, using `checkpoint-epoch4.pth`, and save the outputs to `dataout3` and `dataout2` respectively. The `HYDRA_FULL_ERROR=1` environment variable ensures detailed error messages are shown for troubleshooting.

### Notes:
- Ensure that the device (GPU/CPU) is set correctly; the config uses `device: auto` for automatic detection.
- The script returns the generated audio tensor and sample rate if `single_txt` is provided, which can be captured in code if modified accordingly.
