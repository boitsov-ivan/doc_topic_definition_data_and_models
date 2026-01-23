import os

import hydra

if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
    print("[constants] Initializing Hydra...")
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config")
    config_name = "config"

    hydra.initialize_config_dir(config_dir=config_dir, version_base=None)
    print(f"[constants] Hydra initialized with config: {config_name}")
else:
    print("[constants] Hydra already initialized")


cfg = hydra.compose(config_name="config")
hydra.core.global_hydra.GlobalHydra.instance().clear()
print(f"[constants] Config structure: {list(cfg.keys())}")


DATA_PATH = cfg.data_load.data_path
MODELS_PATH = cfg.model.model_local_path
VOCAB_PATH = cfg.data_load.vocab_path
ONNX_PATH = cfg.model.onnx_path

NUM_CLASSES = cfg.model.num_classes
VAL_PART = cfg.training.val_part
LR = cfg.training.lr
BATCH_SIZE = cfg.training.batch_size
NUM_WORKERS = cfg.training.num_workers

X_LABEL = cfg.data_load.x_label
X_INIT_LABEL = cfg.data_load.x_init_label
Y_LABEL = cfg.data_load.y_label
MAX_PAD_LEN = cfg.data_load.max_pad_len
UNK_TOKEN = cfg.data_load.unk_token

print("[constants] Successfully loaded ALL values from Hydra composition!")
