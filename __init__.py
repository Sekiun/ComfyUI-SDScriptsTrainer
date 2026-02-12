# ComfyUI-SDScriptsTrainer
# ComfyUI custom nodes for sd-scripts LoRA training.

from .nodes import (
    SDScriptsDatasetConfig,
    SDScriptsTrainParams,
    SDScriptsLoRATrain
)
from .preset_api import register_preset_routes

WEB_DIRECTORY = "./web"

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

try:
    register_preset_routes()
except RuntimeError:
    # Running outside of ComfyUI runtime (e.g. during tooling)
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
