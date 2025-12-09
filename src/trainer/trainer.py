from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.logger.utils import plot_waveform
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    def __init__(self, discriminators, *args, **kwargs):
        self.discriminators = discriminators
        super().__init__(*args, **kwargs)

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # does nothing now

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer_gen.zero_grad()
            self.optimizer_disc.zero_grad()
        
        raw_audio = batch["audio"]
        spectrogram = batch["spectrogram"]

        loss_disc = self.criterion["discriminator"]
        loss_gen = self.criterion["generator"]

        # Generator loss
        G_s = self.model(spectrogram)

        batch.update(G_s)
        lst = []
        batch.update(loss_gen(generator=self.model, discriminators=self.discriminators,**batch))
        if self.is_train:
            batch["gan_loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer_gen.step()
            if self.lr_scheduler_gen is not None:
                self.lr_scheduler_gen.step()
            self.optimizer_gen.zero_grad()
            self.optimizer_disc.zero_grad() 

        batch.update(loss_disc(generator=self.model, discriminators=self.discriminators,**batch))

        if self.is_train:
            batch["dis_loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer_disc.step()
            if self.lr_scheduler_disc is not None:
                self.lr_scheduler_disc.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())
        # look at metrics
        #for met in metric_funcs:
        #    metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_waveforms_and_audio(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)
    
    def log_waveforms_and_audio(self, pred_wav, audio, **batch):
        sample_rate = getattr(self.config, 'sample_rate', 22050)  # Adjust based on your config

        pred_image = plot_waveform(pred_wav[0].detach().cpu(), "Predicted Waveform")
        self.writer.add_image("predicted_waveform", pred_image)

        self.writer.add_audio("predicted_audio", pred_wav[0], sample_rate=sample_rate)

        diff = pred_wav[0] - audio[0]
        diff_image = plot_waveform(diff.detach().cpu(), "Difference (Pred - Original)")
        self.writer.add_image("waveform_difference", diff_image)

        self.writer.add_audio("difference_audio", diff, sample_rate=sample_rate)

        if not self.is_train:
            orig_image = plot_waveform(audio[0].detach().cpu(), "Original Audio")
            self.writer.add_image("original_waveform", orig_image)
            self.writer.add_audio("original_audio", audio[0], sample_rate=sample_rate)

