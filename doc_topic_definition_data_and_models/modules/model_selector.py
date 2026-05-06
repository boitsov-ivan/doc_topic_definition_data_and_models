from doc_topic_definition_data_and_models.modules.cnn_bilstm_model import (
    CNNBiLSTMClassifier,
)
from doc_topic_definition_data_and_models.modules.lstm_model import LSTMClassifier
from doc_topic_definition_data_and_models.modules.rnn_model import RNNClassifier




def get_model(vocab_size, conf, num_classes):
    """Model selection for multi-label classification
    
    Args:
        vocab_size (int): размер словаря
        conf (DictConfig): конфигурация
        num_classes (int): количество классов для multi-label классификации
    
    Returns:
        model: PyTorch модель
    """
    
    label = conf["model"]["label"]
    

    if "num_classes" not in conf["model"] or conf["model"]["num_classes"] is None:
        model_num_classes = num_classes
    else:
        model_num_classes = conf["model"]["num_classes"]
    

    input_size = conf["model"].get("input_size", 300)
    hidden_size = conf["model"].get("hidden_size", 256)
    num_layers = conf["model"].get("num_layers", 2)
    dropout = conf["training"].get("dropout", 0.5)
    
    if label == "CNN_BILSTM":
        return CNNBiLSTMClassifier(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=model_num_classes, 
            dropout=dropout,
        )
    
    if label == "LSTM":
        return LSTMClassifier(
            vocab_size=vocab_size,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=model_num_classes, 
            dropout=dropout,
        )
    
    if label == "RNN":
        return RNNClassifier(
            vocab_size=vocab_size,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=model_num_classes,
            dropout=dropout,
        )
    
    
    if label == "TRANSFORMER":
        from doc_topic_definition_data_and_models.modules.transformer_model import (
            TransformerClassifier,
        )
        return TransformerClassifier(
            vocab_size=vocab_size,
            d_model=conf["model"].get("d_model", 256),
            nhead=conf["model"].get("nhead", 8),
            num_layers=conf["model"].get("num_layers", 3),
            num_classes=model_num_classes,
            dropout=dropout,
        )
    
    
    print(f"Model {label} not found, using LSTM as default")
    return LSTMClassifier(
        vocab_size=vocab_size,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=model_num_classes,
        dropout=dropout,
    )