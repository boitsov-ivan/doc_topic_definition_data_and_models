import json
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchtext.vocab import vocab as Vocab

from doc_topic_definition_data_and_models.modules.preprocessing import (
    get_tokenized_sentences,
    pad_num_sentences,
    preprocessing,
    tk,
)


class TextDataset(Dataset):
    """Dataset class for torch dataloaders"""

    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return torch.tensor(self.sentences[index], dtype=torch.long), torch.tensor(
            self.labels[index], dtype=torch.float
        )


def get_top_k_classes(train_df, y_label, num_classes):
    """
    Выбирает top-K самых популярных классов из данных
    
    Args:
        train_df: DataFrame с данными
        y_label: имя колонки с метками классов
        num_classes: количество классов для отбора
    
    Returns:
        top_classes: список top-K классов
        class_to_idx: словарь класса в индекс
        idx_to_class: словарь индекса в класс
    """

    all_classes = []
    for labels_str in train_df[y_label]:
        if isinstance(labels_str, str):
            classes = [c.strip() for c in labels_str.split(',')]
            all_classes.extend(classes)
        elif isinstance(labels_str, list):
            all_classes.extend(labels_str)
    
    class_counts = Counter(all_classes)
    
    top_classes = [cls for cls, _ in class_counts.most_common(num_classes)]
    
    class_to_idx = {cls: idx for idx, cls in enumerate(top_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    return top_classes, class_to_idx, idx_to_class


def encode_multilabel_labels(labels_str, class_to_idx, num_classes):
    """
    Преобразует строку с классами в multi-label бинарный вектор
    
    Args:
        labels_str: строка с классами через запятую
        class_to_idx: словарь映射 класса в индекс
        num_classes: общее количество классов
    
    Returns:
        бинарный numpy array длины num_classes
    """
    label_vector = torch.zeros(num_classes, dtype=torch.float)
    
    if isinstance(labels_str, str) and labels_str:
        classes = [c.strip() for c in labels_str.split(',')]
        for cls in classes:
            if cls in class_to_idx:
                label_vector[class_to_idx[cls]] = 1.0
    
    return label_vector


def get_dataloaders_after_preprocess(
    train_df,
    vocab_path,
    BATCH_SIZE,
    MAX_PAD_LEN,
    VAL_PART,
    X_INIT_LABEL,
    X_LABEL,
    Y_LABEL,
    NUM_CLASSES,
):
    """
    Prepocesses input train data, writes down a vocabulary, makes dataloaders from train data

    Args:
        train_df (pd.DataFrame): train data (to be splitted into train and val)
        vocab_path (str): vocabulary from train data path
        NUM_CLASSES (int): количество топ-классов для отбора

    Returns:
        vocab (dict): vocabulary dictionary
        train_loader (torch.utils.data.DataLoader): dataloader from train
        val_loader (torch.utils.data.DataLoader): dataloader from validation
        class_to_idx (dict): mapping class name to index
        idx_to_class (dict): mapping index to class name
    """

    train_df[X_LABEL] = train_df[X_INIT_LABEL].apply(preprocessing)
    
    top_classes, class_to_idx, idx_to_class = get_top_k_classes(
        train_df, Y_LABEL, NUM_CLASSES
    )
    

    train_df['encoded_labels'] = train_df[Y_LABEL].apply(
        lambda x: encode_multilabel_labels(x, class_to_idx, NUM_CLASSES)
    )
    

    dataset = TextDataset(train_df[X_LABEL].tolist(), train_df['encoded_labels'].tolist())


    val_size = int(VAL_PART * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    x_train = [dataset.sentences[i] for i in train_dataset.indices]
    y_train = [dataset.labels[i] for i in train_dataset.indices]
    
    x_val = [dataset.sentences[i] for i in val_dataset.indices]
    y_val = [dataset.labels[i] for i in val_dataset.indices]


    all_sentences = train_df[X_LABEL].values.tolist()

    token_counter = Counter()
    for tokens in get_tokenized_sentences(all_sentences):
        token_counter.update(tokens)

    special_tokens = ["<unk>"]

    vocab = Vocab(token_counter, specials=special_tokens)

    vocab_dict = vocab.get_stoi()

    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f)
    

    class_mapping_path = vocab_path.replace('.json', '_class_mapping.json')
    with open(class_mapping_path, "w") as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'num_classes': NUM_CLASSES
        }, f)


    x_train_sequences = [
        [vocab_dict.get(token, vocab_dict["<unk>"]) for token in tk(text)] 
        for text in x_train
    ]
    x_val_sequences = [
        [vocab_dict.get(token, vocab_dict["<unk>"]) for token in tk(text)] 
        for text in x_val
    ]


    x_train_padded = [pad_num_sentences(cs, MAX_PAD_LEN) for cs in x_train_sequences]
    x_val_padded = [pad_num_sentences(cs, MAX_PAD_LEN) for cs in x_val_sequences]


    y_train_np = torch.stack(y_train).numpy()
    y_val_np = torch.stack(y_val).numpy()


    train_data = TextDataset(x_train_padded, y_train_np)
    val_data = TextDataset(x_val_padded, y_val_np)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    return vocab, train_loader, val_loader, class_to_idx, idx_to_class


def get_test_dataloader_after_preprocess(
    test_df,
    vocab_path,
    BATCH_SIZE,
    MAX_PAD_LEN,
    VAL_PART,
    X_INIT_LABEL,
    X_LABEL,
    Y_LABEL,
    NUM_CLASSES, 
):
    """
    Prepocesses input test data, makes dataloaders it

    Args:
        test_df (pd.DataFrame): test data
        vocab_path (str): vocabulary from train data path

    Returns:
        test_loader (torch.utils.data.DataLoader): dataloader from test
        class_to_idx (dict): mapping class name to index
        idx_to_class (dict): mapping index to class name
    """


    class_mapping_path = vocab_path.replace('.json', '_class_mapping.json')
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)
    
    class_to_idx = class_mapping['class_to_idx']
    idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}
    NUM_CLASSES = class_mapping['num_classes']


    test_df[X_LABEL] = test_df[X_INIT_LABEL].apply(preprocessing)
    

    test_df['encoded_labels'] = test_df[Y_LABEL].apply(
        lambda x: encode_multilabel_labels(x, class_to_idx, NUM_CLASSES)
    )
    
    test_dataset = TextDataset(test_df[X_LABEL].tolist(), test_df['encoded_labels'].tolist())

    with open(vocab_path, "r") as f:
        vocab_dict = json.load(f)

    x_test_sequences = [
        [vocab_dict.get(token, vocab_dict["<unk>"]) for token in tk(text)]
        for text in test_dataset.sentences
    ]

    x_test_padded = [pad_num_sentences(cs, MAX_PAD_LEN) for cs in x_test_sequences]

    y_test_np = torch.stack(test_dataset.labels).numpy()

    test_data = TextDataset(x_test_padded, y_test_np)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return len(vocab_dict), test_loader, class_to_idx, idx_to_class