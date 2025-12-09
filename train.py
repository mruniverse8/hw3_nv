import warnings

import hydra
import torch
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf
from itertools import chain  # lazy

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import (
    set_random_seed,
    setup_saving_and_logging,
    get_available_devices,
    setup_data_parallel,
)

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

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    # Device setup with parallel option
    parallel_option = config.trainer.get("parallel_option", False)
    config_device_ids = config.trainer.get("device_ids", None)
    device_ids = None

    if config.trainer.device == "auto":
        if parallel_option and torch.cuda.is_available():
            device, detected_device_ids, use_parallel = get_available_devices()
            # Use config device_ids if specified, otherwise use detected
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

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    discriminators = torch.nn.ModuleList(instantiate(config.discriminators)).to(device)

    # Wrap with DataParallel if enabled
    if device_ids is not None and len(device_ids) >= 2:
        model = setup_data_parallel(model, device_ids)
        # Wrap each discriminator individually
        discriminators = torch.nn.ModuleList([
            setup_data_parallel(disc, device_ids) for disc in discriminators
        ])
        logger.info(f"Models wrapped with DataParallel on GPUs: {device_ids}")

    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function) # loss function is a dict 
    loss_function["discriminator"].to(device)
    loss_function["generator"].to(device)
    metrics = {"train": [], "inference": []}
    metrics = instantiate(config.metrics) # simple?

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer_cls = get_class(config.optimizer.cls)
    project_config["optimizer"].pop("cls")
    optimizer_discriminator = optimizer_cls(
        chain(*[disc.parameters() for disc in discriminators]),
        **project_config["optimizer"]
    )
    optimizer_generator = optimizer_cls(
        model.parameters(), **project_config["optimizer"]
    )
    
    lr_scheduler_gen = instantiate(config.lr_scheduler, optimizer=optimizer_generator)
    lr_scheduler_disc = instantiate(config.lr_scheduler, optimizer=optimizer_discriminator)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        discriminators=discriminators,
        criterion=loss_function,
        metrics=metrics,
        optimizer_discriminator=optimizer_discriminator,
        optimizer_generator=optimizer_generator,
        lr_scheduler_gen=lr_scheduler_gen,
        lr_scheduler_disc=lr_scheduler_disc,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
