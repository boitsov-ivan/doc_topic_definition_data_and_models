import fire
import pandas as pd
import pytorch_lightning as pl
import torch
import json
import os

import doc_topic_definition_data_and_models.modules.constants as constants
from doc_topic_definition_data_and_models.modules.dataloaders import (
    get_test_dataloader_after_preprocess,
)
from doc_topic_definition_data_and_models.modules.trainer import TextClassifier


def main(test_dir: str, checkpoint_name: str, output_predictions: str = None) -> None:
    """
    Запуск инференса на тестовых данных для multi-label классификации
    
    Args:
        test_dir: имя папки с тестовыми данными
        checkpoint_name: имя файла чекпоинта модели
        output_predictions: опциональный путь для сохранения предсказаний
    """
    
    test_csv = f"{constants.DATA_PATH}/{test_dir}"
    test_df = pd.read_csv(test_csv)
    
    print(f"Loading test data from: {test_csv}")
    print(f"Test data shape: {test_df.shape}")
    
    class_mapping_path = constants.VOCAB_PATH.replace('.json', '_class_mapping.json')

    if not os.path.exists(class_mapping_path):
        checkpoint_dir = os.path.dirname(checkpoint_name)
        class_mapping_path = os.path.join(checkpoint_dir, "class_mapping.json")
        
        if not os.path.exists(class_mapping_path):
            class_mapping_path = os.path.join(
                constants.MODELS_PATH, 
                "class_mapping.json"
            )
    
    print(f"Loading class mapping from: {class_mapping_path}")
    
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)
    
    class_to_idx = class_mapping['class_to_idx']
    idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}
    num_classes = class_mapping['num_classes']
    
    print(f"Number of classes: {num_classes}")
    print(f"Top classes: {list(class_to_idx.keys())[:5]}...")
    
    
    
    vocab_size, test_loader = get_test_dataloader_after_preprocess(
        test_df,
        constants.VOCAB_PATH,
        constants.BATCH_SIZE,
        constants.MAX_PAD_LEN,
        constants.VAL_PART,
        constants.X_INIT_LABEL,
        constants.X_LABEL,
        constants.Y_LABEL,
        constants.NUM_CLASSES, 
    )
    
    
    checkpoint_path = f"{constants.MODELS_PATH}/{checkpoint_name}"
    print(f"Loading model from: {checkpoint_path}")
    
    module = TextClassifier.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        map_location=None  
    )
    
    module.eval()
    
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )
    

    print("\nRunning inference...")
    test_results = trainer.test(module, dataloaders=test_loader, verbose=True)
    
    print("\n" + "="*50)
    print("TEST RESULTS:")
    print("="*50)
    print(f"Test Loss: {test_results[0]['test_loss']:.4f}")
    print(f"Test Accuracy: {test_results[0]['test_acc']:.4f}")
    print(f"Test F1-Score (macro): {test_results[0]['test_f1']:.4f}")
    if 'test_precision' in test_results[0]:
        print(f"Test Precision (macro): {test_results[0]['test_precision']:.4f}")
    if 'test_recall' in test_results[0]:
        print(f"Test Recall (macro): {test_results[0]['test_recall']:.4f}")
    

    if output_predictions:
        print(f"\nSaving predictions to: {output_predictions}")
        predictions_df = get_predictions(module, test_loader, idx_to_class, test_df)
        predictions_df.to_csv(output_predictions, index=False)
        print(f"Predictions saved successfully!")
        print(f"Predictions shape: {predictions_df.shape}")
    
    return test_results[0]


def get_predictions(model, test_loader, idx_to_class, original_df=None, threshold=0.5):
    """
    Генерирует предсказания для тестовых данных
    
    Args:
        model: обученная модель
        test_loader: DataLoader с тестовыми данными
        idx_to_class: словарь из индекса в название класса
        original_df: оригинальный DataFrame (опционально, для добавления предсказаний)
        threshold: порог для бинаризации предсказаний
    
    Returns:
        DataFrame с предсказаниями
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            texts, labels = batch
            outputs = model(texts)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > threshold).int()
            
            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            
            
            batch_indices = list(range(batch_idx * test_loader.batch_size, 
                                      batch_idx * test_loader.batch_size + len(texts)))
            all_indices.extend(batch_indices)
    
    
    predicted_classes_list = []
    for pred in all_predictions:
        classes = [idx_to_class[i] for i, p in enumerate(pred) if p == 1]
        predicted_classes_list.append(classes)
    

    proba_df = pd.DataFrame(
        all_probabilities,
        columns=[f"prob_{idx_to_class[i]}" for i in range(len(idx_to_class))]
    )
    
    if original_df is not None:
        results_df = original_df.copy()
        results_df['predicted_classes'] = predicted_classes_list
        results_df['num_predicted_classes'] = [len(p) for p in predicted_classes_list]
    else:
        results_df = pd.DataFrame({
            'predicted_classes': predicted_classes_list,
            'num_predicted_classes': [len(p) for p in predicted_classes_list]
        })
    
    
    for i, class_name in idx_to_class.items():
        results_df[f'pred_{class_name}'] = [pred[i] for pred in all_predictions]
    
    
    results_df['max_probability'] = [max(probs) for probs in all_probabilities]
    results_df['mean_probability'] = [sum(probs)/len(probs) for probs in all_probabilities]
    
    return results_df


def predict_single_text(model, text, vocab_dict, idx_to_class, max_pad_len, threshold=0.5):
    """
    Предсказание для одного текста
    
    Args:
        model: обученная модель
        text: текст для классификации
        vocab_dict: словарь токен -> индекс
        idx_to_class: словарь из индекса в название класса
        max_pad_len: максимальная длина последовательности
        threshold: порог для бинаризации
    
    Returns:
        dict с предсказаниями
    """
    from doc_topic_definition_data_and_models.modules.preprocessing import tk, pad_num_sentences
    
    model.eval()
    
    
    tokens = tk(text)
    indices = [vocab_dict.get(token, vocab_dict['<unk>']) for token in tokens]
    
    
    padded = pad_num_sentences(indices, max_pad_len)
    

    input_tensor = torch.tensor([padded], dtype=torch.long)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).int()
    
    proba_np = probabilities.cpu().numpy()[0]
    pred_np = predictions.cpu().numpy()[0]
    
    predicted_classes = [idx_to_class[i] for i, p in enumerate(pred_np) if p == 1]
    

    class_proba = {idx_to_class[i]: proba_np[i] for i in range(len(idx_to_class))}
    
    return {
        'text': text,
        'predicted_classes': predicted_classes,
        'probabilities': class_proba,
        'top_class': max(class_proba, key=class_proba.get) if predicted_classes else None,
        'top_probability': max(proba_np) if len(proba_np) > 0 else 0
    }


if __name__ == "__main__":
    fire.Fire(main)