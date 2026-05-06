import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import F1Score, Precision, Recall, AUROC


class TextClassifier(pl.LightningModule):
    """
    Module for training and evaluation models
    for multi-label text classification task.
    """

    def __init__(self, model, lr, vocab_size, dropout, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.num_classes = num_classes

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_f1 = F1Score(
            task="multilabel", num_labels=self.num_classes, average="macro"
        )
        self.val_f1 = F1Score(
            task="multilabel", num_labels=self.num_classes, average="macro"
        )
        self.test_f1 = F1Score(
            task="multilabel", num_labels=self.num_classes, average="macro"
        )
        
        self.train_precision = Precision(
            task="multilabel", num_labels=self.num_classes, average="macro"
        )
        self.val_precision = Precision(
            task="multilabel", num_labels=self.num_classes, average="macro"
        )
        self.train_recall = Recall(
            task="multilabel", num_labels=self.num_classes, average="macro"
        )
        self.val_recall = Recall(
            task="multilabel", num_labels=self.num_classes, average="macro"
        )
        

        self.train_auroc = AUROC(
            task="multilabel", num_labels=self.num_classes, average="macro"
        )
        self.val_auroc = AUROC(
            task="multilabel", num_labels=self.num_classes, average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        texts, labels = batch
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)
        
        preds = torch.sigmoid(outputs) > 0.5
        
        self.train_f1.update(preds, labels.int())
        self.train_precision.update(preds, labels.int())
        self.train_recall.update(preds, labels.int())
        self.train_auroc.update(outputs, labels.int())
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def on_train_epoch_end(self):
        """Вычисляем метрики в конце эпохи обучения"""
        train_f1_score = self.train_f1.compute()
        train_precision = self.train_precision.compute()
        train_recall = self.train_recall.compute()
        train_auroc = self.train_auroc.compute()
        
        self.log("train_f1", train_f1_score, prog_bar=True)
        self.log("train_precision", train_precision, prog_bar=False)
        self.log("train_recall", train_recall, prog_bar=False)
        self.log("train_auroc", train_auroc, prog_bar=False)
        
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_auroc.reset()

    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)
        
        preds = torch.sigmoid(outputs) > 0.5
        
        acc = (preds == labels.bool()).float().mean()
        
        self.val_f1.update(preds, labels.int())
        self.val_precision.update(preds, labels.int())
        self.val_recall.update(preds, labels.int())
        self.val_auroc.update(outputs, labels.int())
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        """Вычисляем метрики в конце эпохи валидации"""
        val_f1_score = self.val_f1.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        val_auroc = self.val_auroc.compute()
        
        self.log("val_f1", val_f1_score, prog_bar=True)
        self.log("val_precision", val_precision, prog_bar=False)
        self.log("val_recall", val_recall, prog_bar=False)
        self.log("val_auroc", val_auroc, prog_bar=False)
        
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_auroc.reset()

    def test_step(self, batch, batch_idx):
        texts, labels = batch
        outputs = self(texts)
        test_loss = self.loss_fn(outputs, labels)
        
        
        preds = torch.sigmoid(outputs) > 0.5
        
        
        acc = (preds == labels.bool()).float().mean()
        
        
        self.test_f1.update(preds, labels.int())
        
        self.log("test_loss", test_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return {
            "test_loss": test_loss,
            "test_acc": acc,
            "preds": preds,
            "labels": labels,
            "raw_outputs": outputs,
        }

    def on_test_epoch_end(self):
        """Вычисляем метрики в конце тестирования"""
        test_f1_score = self.test_f1.compute()
        self.log("test_f1", test_f1_score, prog_bar=True)
        
        self.test_f1_score = test_f1_score
        self.test_f1.reset()
        

        self.log("test_f1_final", test_f1_score)
        print(f"\nTest Results - F1 Score (macro): {test_f1_score:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """Метод для инференса"""
        texts, _ = batch 
        outputs = self(texts)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).int()
        return {
            "probabilities": probabilities,
            "predictions": predictions,
            "logits": outputs
        }