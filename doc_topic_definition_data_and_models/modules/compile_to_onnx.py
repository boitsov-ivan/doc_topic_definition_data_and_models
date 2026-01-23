import fire
import numpy as np
import onnxruntime
import torch

import classifier_of_press_releases_cbrf.modules.constants as constants
from classifier_of_press_releases_cbrf.modules.trainer import TextClassifier


def main(checkpoint_name: str) -> None:
    module = TextClassifier.load_from_checkpoint(
        checkpoint_path=f"{constants.MODELS_PATH}/{checkpoint_name}",
        num_classes=constants.NUM_CLASSES,
    )
    print("Successfully loaded from checkpoint")
    module.eval()

    vocab_size = module.hparams.vocab_size
    input_array = torch.randint(0, vocab_size, (1, 90))

    module.to_onnx(
        constants.ONNX_PATH,
        input_array,
        export_params=True,
        input_names=["PREPROCESSED_TEXT"],
        output_names=["LOGITS"],
        dynamic_axes={
            "PREPROCESSED_TEXT": {0: "batch_size", 1: "seq_length"},
            "LOGITS": {0: "batch_size"},
        },
    )

    print("Model compiled to ONNX")

    ort_session = onnxruntime.InferenceSession(
        constants.ONNX_PATH, providers=["CPUExecutionProvider"]
    )
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_array.numpy().astype(np.int64)}
    ort_outs = ort_session.run(None, ort_inputs)

    if ort_outs:
        print("ONNX model check passed")


if __name__ == "__main__":
    fire.Fire(main)
