import fire
import pandas as pd
import pytorch_lightning as pl

import classifier_of_press_releases_cbrf.modules.constants as constants
from classifier_of_press_releases_cbrf.modules.dataloaders import (
    get_test_dataloader_after_preprocess,
)
from classifier_of_press_releases_cbrf.modules.trainer import TextClassifier


def main(test_dir: str, checkpoint_name: str) -> None:
    test_csv = f"{constants.DATA_PATH}/{test_dir}"
    test_df = pd.read_csv(test_csv)

    _, test_loader = get_test_dataloader_after_preprocess(
        test_df,
        constants.VOCAB_PATH,
        constants.BATCH_SIZE,
        constants.MAX_PAD_LEN,
        constants.VAL_PART,
        constants.X_INIT_LABEL,
        constants.X_LABEL,
        constants.Y_LABEL,
    )
    module = TextClassifier.load_from_checkpoint(
        checkpoint_path=f"{constants.MODELS_PATH}/{checkpoint_name}",
        num_classes=constants.NUM_CLASSES,
    )

    module.eval()

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    test_results = trainer.test(module, dataloaders=test_loader)
    print(f"Final Test Accuracy: {test_results[0]['test_acc']:.4f}")
    print(f"Final Test F1-Score: {test_results[0]['test_f1']:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
