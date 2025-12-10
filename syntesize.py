import warnings

import hydra
import torch
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf
from itertools import chain  # lazy
from src.trainer import Inferencer

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import (
    set_random_seed,
    setup_saving_and_logging,
    get_available_devices,
    setup_data_parallel,
)
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    #project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    #writer = instantiate(config.writer, logger, project_config)


    metrics = {"train": [], "inference": []}
    metrics = instantiate(config.metrics) # simple?

    parallel_option = config.trainer.get("parallel_option", False)
    config_device_ids = config.trainer.get("device_ids", None)
    device_ids = None

    if config.trainer.device == "auto":
        if parallel_option and torch.cuda.is_available():
            device, detected_device_ids, use_parallel = get_available_devices()
            device_ids = config_device_ids if config_device_ids is not None else detected_device_ids
            if device_ids and len(device_ids) >= 2:
                logger.info(f"Multi-GPU training enabled on devices: {device_ids}")
            else:
                device_ids = None
                logger.info("Single GPU training (not enough GPUs for parallel)")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device
    
    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    discriminators = torch.nn.ModuleList(instantiate(config.discriminators)).to(device)

    if device_ids is not None and len(device_ids) >= 2:
        model = setup_data_parallel(model, device_ids)
        discriminators = torch.nn.ModuleList([
            setup_data_parallel(disc, device_ids) for disc in discriminators
        ])
        logger.info(f"Models wrapped with DataParallel on GPUs: {device_ids}")

    logger.info(model)

    # save_path for model predictions
    save_path = ROOT_PATH / "inferece_data" / "saved" / config.trainer.save_dir
    save_path.mkdir(exist_ok=True, parents=True)
    inferencer = Inferencer(
        model=model,
        discriminators=discriminators,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )
    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")



if __name__ == "__main__":
    main()
