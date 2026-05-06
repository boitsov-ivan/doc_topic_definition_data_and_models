import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import ast

import doc_topic_definition_data_and_models.modules.constants as constants
from doc_topic_definition_data_and_models.modules.dataloaders import (
    get_dataloaders_after_preprocess,
)
from doc_topic_definition_data_and_models.modules.model_selector import get_model
from doc_topic_definition_data_and_models.modules.trainer import TextClassifier


def parse_labels(label_str):
    if pd.isna(label_str):
        return []
    if isinstance(label_str, str):
        try:
            return ast.literal_eval(label_str)
        except:
            return [x.strip() for x in label_str.strip('[]').split(',') if x.strip()]
    return label_str


def safe_stratified_split(df, test_size=0.2, random_state=42, label_col='topics'):
    df_clean = df.copy()
    df_clean['parsed_labels'] = df_clean[label_col].apply(parse_labels)
    df_clean['primary_label'] = df_clean['parsed_labels'].apply(lambda x: x[0] if x else 'unknown')
    
    class_counts = df_clean['primary_label'].value_counts()
    rare_classes = class_counts[class_counts < 3].index.tolist()
    
    if len(rare_classes) > 0:
        print(f"\nWarning: Found rare classes with <3 samples: {len(rare_classes)} classes")
        
        rare_df = df_clean[df_clean['primary_label'].isin(rare_classes)]
        common_df = df_clean[~df_clean['primary_label'].isin(rare_classes)]
        
        print(f"Samples with rare classes: {len(rare_df)}")
        print(f"Samples with common classes: {len(common_df)}")
        
        if len(common_df) > 0:
            try:
                train_common, test_common = train_test_split(
                    common_df,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=common_df['primary_label']
                )
            except ValueError as e:
                print(f"Stratified split failed, using simple split: {e}")
                train_common, test_common = train_test_split(
                    common_df,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=None
                )
            
            if len(rare_df) > 0:
                if len(rare_df) >= 2:
                    train_rare, test_rare = train_test_split(
                        rare_df,
                        test_size=min(test_size, 0.5),
                        random_state=random_state,
                        stratify=None
                    )
                else:
                    train_rare = rare_df
                    test_rare = pd.DataFrame(columns=rare_df.columns)
                
                train_df = pd.concat([train_common, train_rare], ignore_index=True)
                test_df = pd.concat([test_common, test_rare], ignore_index=True)
            else:
                train_df = train_common
                test_df = test_common
        else:
            print("All classes are rare, using simple random split")
            train_df, test_df = train_test_split(
                df_clean,
                test_size=test_size,
                random_state=random_state,
                stratify=None
            )
    else:
        train_df, test_df = train_test_split(
            df_clean,
            test_size=test_size,
            random_state=random_state,
            stratify=df_clean['primary_label']
        )
    
    train_df = train_df.drop(columns=['parsed_labels', 'primary_label'])
    test_df = test_df.drop(columns=['parsed_labels', 'primary_label'])
    
    return train_df, test_df


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    full_data_path = "../../data/docs_cleaned.csv"
    print(f"Loading full data from: {full_data_path}")
    full_df = pd.read_csv(full_data_path)
    print(f"Total samples: {len(full_df)}")
    
    if constants.Y_LABEL not in full_df.columns:
        raise ValueError(f"Column '{constants.Y_LABEL}' not found in data. Available columns: {full_df.columns.tolist()}")
    
    print("\nParsing labels...")
    full_df['parsed_labels'] = full_df[constants.Y_LABEL].apply(parse_labels)
    full_df['primary_label'] = full_df['parsed_labels'].apply(lambda x: x[0] if x else 'unknown')
    
    print("\n" + "="*50)
    print("Class distribution analysis:")
    print("="*50)
    class_counts = full_df['primary_label'].value_counts()
    print(f"Total unique classes: {len(class_counts)}")
    print(f"Classes with <3 samples: {(class_counts < 3).sum()}")
    print(f"Classes with <10 samples: {(class_counts < 10).sum()}")
    print(f"Most common class: {class_counts.index[0]} ({class_counts.iloc[0]} samples)")
    
    print("\nTop 15 classes by frequency:")
    for i, (cls, count) in enumerate(class_counts.head(15).items()):
        cls_str = str(cls)[:50]
        print(f"  {i+1}. {cls_str}: {count} samples ({count/len(full_df)*100:.1f}%)")
    
    print("\n" + "="*50)
    print("Splitting data into train (80%) and test (20%)...")
    print("="*50)
    
    train_df, test_df = safe_stratified_split(
        full_df,
        test_size=0.2,
        random_state=42,
        label_col=constants.Y_LABEL
    )
    
    print(f"\nSplit results:")
    print(f"  Train size: {len(train_df)} ({len(train_df)/len(full_df)*100:.1f}%)")
    print(f"  Test size: {len(test_df)} ({len(test_df)/len(full_df)*100:.1f}%)")
    
    train_save_path = config["data_load"]["train_data_path"]
    test_save_path = config["data_load"].get("test_data_path", "../../data/docs_test.csv")
    
    os.makedirs(os.path.dirname(train_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_save_path), exist_ok=True)
    
    train_df.to_csv(train_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)
    
    print(f"\nSaved files:")
    print(f"  Train: {train_save_path}")
    print(f"  Test: {test_save_path}")
    
    print("\n" + "="*50)
    print("Train set class distribution (top 15):")
    print("="*50)
    train_parsed = train_df[constants.Y_LABEL].apply(parse_labels)
    train_primary = train_parsed.apply(lambda x: x[0] if x else 'unknown')
    train_class_counts = train_primary.value_counts()
    for i, (cls, count) in enumerate(train_class_counts.head(15).items()):
        cls_str = str(cls)[:50]
        print(f"  {i+1}. {cls_str}: {count} ({count/len(train_df)*100:.1f}%)")
    
    print("\n" + "="*50)
    print("Preparing dataloaders from train data...")
    print("="*50)
    
    vocab, train_loader, val_loader, class_to_idx, idx_to_class = get_dataloaders_after_preprocess(
        train_df,
        config["data_load"]["vocab_path"],
        constants.BATCH_SIZE,
        constants.MAX_PAD_LEN,
        constants.VAL_PART,
        constants.X_INIT_LABEL,
        constants.X_LABEL,
        constants.Y_LABEL,
        constants.NUM_CLASSES,
    )
    
    print(f"\nVocabulary size: {len(vocab.get_stoi()) + 1}")
    print(f"Number of classes (top-{constants.NUM_CLASSES}): {len(class_to_idx)}")
    print(f"Selected classes: {list(class_to_idx.keys())[:10]}")
    
    loggers = [
        pl.loggers.WandbLogger(
            project=config["logging"]["project"],
            name=config["logging"]["name"],
            save_dir=config["logging"]["save_dir"],
        )
    ]
    
    loggers[0].experiment.config.update({
        "total_samples": len(full_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "val_samples": int(len(train_df) * constants.VAL_PART),
        "num_total_classes": len(class_counts),
        "num_selected_classes": constants.NUM_CLASSES,
        "selected_classes": list(class_to_idx.keys()),
        "random_state": 42,
    })
    
    vocab_size = len(vocab.get_stoi()) + 1
    
    model = get_model(vocab_size, config, num_classes=constants.NUM_CLASSES)
    module = TextClassifier(
        model,
        lr=config["training"]["lr"],
        vocab_size=vocab_size,
        dropout=config["training"]["dropout"],
        num_classes=constants.NUM_CLASSES,
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
    
    class SaveClassMappingCallback(pl.Callback):
        def on_train_end(self, trainer, pl_module):
            import json
            checkpoint_dir = trainer.checkpoint_callback.dirpath
            class_mapping_path = os.path.join(checkpoint_dir, "class_mapping.json")
            
            with open(class_mapping_path, "w") as f:
                json.dump({
                    'class_to_idx': class_to_idx,
                    'idx_to_class': idx_to_class,
                    'num_classes': constants.NUM_CLASSES
                }, f, indent=2)
            
            print(f"\nClass mapping saved to {class_mapping_path}")
    
    callbacks.append(SaveClassMappingCallback())
    
    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {constants.BATCH_SIZE}")
    print(f"Max epochs: {config['training']['num_epochs']}")
    print(f"Learning rate: {config['training']['lr']}")
    
    trainer.fit(module, train_loader, val_loader)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    if trainer.checkpoint_callback.best_model_path:
        print(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    
    split_info_path = os.path.join(config["model"]["model_local_path"], "split_info.txt")
    os.makedirs(os.path.dirname(split_info_path), exist_ok=True)
    with open(split_info_path, "w") as f:
        f.write("DATA SPLIT INFORMATION\n")
        f.write("="*40 + "\n")
        f.write(f"Total samples: {len(full_df)}\n")
        f.write(f"Train samples: {len(train_df)} ({len(train_df)/len(full_df)*100:.1f}%)\n")
        f.write(f"Test samples: {len(test_df)} ({len(test_df)/len(full_df)*100:.1f}%)\n")
        f.write(f"Validation samples: {int(len(train_df) * constants.VAL_PART)} ({constants.VAL_PART*100:.0f}% of train)\n")
        f.write(f"Random state: 42\n")
        f.write(f"Stratification: Adaptive by primary class\n")
        f.write(f"Total unique classes: {len(class_counts)}\n")
        f.write(f"Selected classes (top-{constants.NUM_CLASSES}): {len(class_to_idx)}\n")
    
    print(f"Split info saved to: {split_info_path}")


if __name__ == "__main__":
    main()