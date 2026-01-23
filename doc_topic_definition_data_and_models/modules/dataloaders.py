import json
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchtext.vocab import vocab as Vocab

from classifier_of_press_releases_cbrf.modules.preprocessing import (
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
            self.labels[index], dtype=torch.long
        )


def get_dataloaders_after_preprocess(
    train_df,
    vocab_path,
    BATCH_SIZE,
    MAX_PAD_LEN,
    VAL_PART,
    X_INIT_LABEL,
    X_LABEL,
    Y_LABEL,
):
    """
    Prepocesses input train data, writes down a vocabulary, makes dataloaders from train data

    Args:
        train_df (pd.DataFrame): train data (to be splitted into train and val)
        vocab_path (str): vocabulary from train data path

    Returns:
        vocab (dict): vocabulary dictionary
        train_loader (torch.utils.data.DataLoader): dataloader from train
        val_loader (torch.utils.data.DataLoader): dataloader from validation
    """

    train_df[X_LABEL] = train_df[X_INIT_LABEL].apply(preprocessing)

    dataset = TextDataset(train_df[X_LABEL], train_df[Y_LABEL])

    val_size = int(VAL_PART * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    x_train = pd.DataFrame(
        {X_LABEL: [dataset.sentences[i] for i in train_dataset.indices]}
    )
    y_train = pd.DataFrame({Y_LABEL: dataset.labels[train_dataset.indices]})

    x_val = pd.DataFrame({X_LABEL: [dataset.sentences[i] for i in val_dataset.indices]})
    y_val = pd.DataFrame({Y_LABEL: dataset.labels[val_dataset.indices]})

    x_train.reset_index(drop=True, inplace=True)
    x_val.reset_index(drop=True, inplace=True)

    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    all_sentences = train_df["preprocessed_sentences"].values.tolist()

    token_counter = Counter()
    for tokens in get_tokenized_sentences(all_sentences):
        token_counter.update(tokens)

    special_tokens = ["<unk>"]

    vocab = Vocab(token_counter, specials=special_tokens)

    stoi_dict = vocab.get_stoi()
    stoi_dict.setdefault("<unk>", None)

    vocab_dict = vocab.get_stoi()

    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f)

    x_train_sequences = [
        [vocab_dict[token] for token in tk(text)] for text in x_train[X_LABEL]
    ]
    x_val_sequences = [
        [vocab_dict[token] for token in tk(text)] for text in x_val[X_LABEL]
    ]

    x_train_padded = [pad_num_sentences(cs, MAX_PAD_LEN) for cs in x_train_sequences]
    x_val_padded = [pad_num_sentences(cs, MAX_PAD_LEN) for cs in x_val_sequences]

    y_train_np = y_train[Y_LABEL].values.astype("int64").flatten()
    y_val_np = y_val[Y_LABEL].values.astype("int64").flatten()

    max_of_y = max(set(y_train_np))

    y_train_np[y_train_np == max_of_y] = 0
    y_val_np[y_val_np == max_of_y] = 0

    train_data = TextDataset(x_train_padded, y_train_np)
    val_data = TextDataset(x_val_padded, y_val_np)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    return vocab, train_loader, val_loader


def get_test_dataloader_after_preprocess(
    test_df,
    vocab_path,
    BATCH_SIZE,
    MAX_PAD_LEN,
    VAL_PART,
    X_INIT_LABEL,
    X_LABEL,
    Y_LABEL,
):
    """
    Prepocesses input test data, makes dataloaders it

    Args:
        test_df (pd.DataFrame): test data
        vocab_path (str): vocabulary from train data path

    Returns:
        test_loader (torch.utils.data.DataLoader): dataloader from test
    """

    test_df[X_LABEL] = test_df[X_INIT_LABEL].apply(preprocessing)

    test_dataset = TextDataset(test_df[X_LABEL], test_df[Y_LABEL])

    x_test = pd.DataFrame({X_LABEL: test_dataset.sentences})
    y_test = pd.DataFrame({Y_LABEL: test_dataset.labels})

    x_test.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)

    y_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    f = open(vocab_path)

    vocab_dict = json.load(f)

    x_test_sequences = [
        [vocab_dict.get(token, vocab_dict["<unk>"]) for token in tk(text)]
        for text in x_test[X_LABEL]
    ]

    x_test_padded = [pad_num_sentences(cs, MAX_PAD_LEN) for cs in x_test_sequences]

    y_test_np = y_test[Y_LABEL].values.astype("int64").flatten()

    max_of_y = max(set(y_test_np))

    y_test_np[y_test_np == max_of_y] = 0

    test_data = TextDataset(x_test_padded, y_test_np)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    return len(vocab_dict), test_loader
