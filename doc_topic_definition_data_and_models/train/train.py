import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

import classifier_of_press_releases_cbrf.modules.constants as constants
from classifier_of_press_releases_cbrf.modules.dataloaders import (
    get_dataloaders_after_preprocess,
)
from classifier_of_press_releases_cbrf.modules.model_selector import get_model
from classifier_of_press_releases_cbrf.modules.trainer import TextClassifier


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    train_df = pd.read_csv(config["data_load"]["train_data_path"])

    vocab, train_loader, val_loader = get_dataloaders_after_preprocess(
        train_df,
        config["data_load"]["vocab_path"],
        constants.BATCH_SIZE,
        constants.MAX_PAD_LEN,
        constants.VAL_PART,
        constants.X_INIT_LABEL,
        constants.X_LABEL,
        constants.Y_LABEL,
    )

    vocab_size = len(vocab.get_stoi()) + 1

    loggers = [
        pl.loggers.WandbLogger(
            project=config["logging"]["project"],
            name=config["logging"]["name"],
            save_dir=config["logging"]["save_dir"],
        )
    ]

    model = get_model(vocab_size, config)
    module = TextClassifier(
        model,
        lr=config["training"]["lr"],
        vocab_size=vocab_size,
        dropout=config["training"]["dropout"],
    )

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]

    callbacks.append(
        ModelCheckpoint(
            dirpath=config["model"]["model_local_path"],
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=config["model"]["save_top_k"],
            every_n_epochs=config["model"]["every_n_epochs"],
        )
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
