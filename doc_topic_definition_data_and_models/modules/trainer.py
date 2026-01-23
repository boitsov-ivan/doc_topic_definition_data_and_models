import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import F1Score


class TextClassifier(pl.LightningModule):
    """
    Module for training and evaluation models
    for text classification task.
    """

    def __init__(self, model, lr, vocab_size, dropout, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_f1 = F1Score(
            task="multiclass", num_classes=self.num_classes, average="macro"
        )

        self.val_f1 = F1Score(
            task="multiclass", num_classes=self.num_classes, average="macro"
        )

        self.test_f1 = F1Score(
            task="multiclass", num_classes=self.num_classes, average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        texts, labels = batch
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        self.train_f1.update(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """Вычисляем F1 в конце эпохи обучения"""
        train_f1_score = self.train_f1.compute()
        self.log("train_f1", train_f1_score, prog_bar=True)
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        acc = (preds == labels).float().mean()

        self.val_f1.update(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        """Вычисляем F1 в конце эпохи валидации"""
        val_f1_score = self.val_f1.compute()
        self.log("val_f1", val_f1_score, prog_bar=True)
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        texts, labels = batch
        outputs = self(texts)
        test_loss = self.loss_fn(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        acc = (preds == labels).float().mean()

        self.test_f1.update(preds, labels)

        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return {
            "test_loss": test_loss,
            "test_acc": acc,
            "preds": preds,
            "labels": labels,
        }

    def on_test_epoch_end(self):
        """Вычисляем F1 в конце тестирования"""
        test_f1_score = self.test_f1.compute()
        self.log("test_f1", test_f1_score, prog_bar=True)

        self.test_f1_score = test_f1_score
        self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=0.5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
