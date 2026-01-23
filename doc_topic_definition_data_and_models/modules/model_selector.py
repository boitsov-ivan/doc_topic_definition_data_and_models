from classifier_of_press_releases_cbrf.modules.cnn_bilstm_model import (
    CNNBiLSTMClassifier,
)
from classifier_of_press_releases_cbrf.modules.lstm_model import LSTMClassifier
from classifier_of_press_releases_cbrf.modules.rnn_model import RNNClassifier


def get_model(vocab_size, conf):
    """Model selection"""

    label = conf["model"]["label"]

    if label == "CNN_BILSTM":
        return CNNBiLSTMClassifier(
            vocab_size=vocab_size,
            # input_size=conf["model"]["input_size"],
            hidden_size=conf["model"]["hidden_size"],
            num_layers=conf["model"]["num_layers"],
            num_classes=conf["model"]["num_classes"],
            dropout=conf["training"]["dropout"],
        )

    if label == "LSTM":
        return LSTMClassifier(
            vocab_size=vocab_size,
            input_size=conf["model"]["input_size"],
            hidden_size=conf["model"]["hidden_size"],
            num_layers=conf["model"]["num_layers"],
            num_classes=conf["model"]["num_classes"],
            dropout=conf["training"]["dropout"],
        )

    if label == "RNN":
        return RNNClassifier(
            vocab_size=vocab_size,
            input_size=conf["model"]["input_size"],
            hidden_size=conf["model"]["hidden_size"],
            num_layers=conf["model"]["num_layers"],
            num_classes=conf["model"]["num_classes"],
            dropout=conf["training"]["dropout"],
        )
