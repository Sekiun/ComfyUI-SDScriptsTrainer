# ComfyUI-SDScriptsTrainer
# ComfyUI custom nodes for sd-scripts LoRA training.

from .nodes import (
    SDScriptsDatasetConfig,
    SDScriptsTrainParams,
    SDScriptsLoRATrain
)

NODE_CLASS_MAPPINGS = {
    "SDScriptsDatasetConfig": SDScriptsDatasetConfig,
    "SDScriptsTrainParams": SDScriptsTrainParams,
    "SDScriptsLoRATrain": SDScriptsLoRATrain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDScriptsDatasetConfig": "SD-Scripts Dataset Config",
    "SDScriptsTrainParams": "SD-Scripts Train Params",
    "SDScriptsLoRATrain": "SD-Scripts LoRA Train",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
