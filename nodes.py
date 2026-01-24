"""
SD-Scripts LoRA training nodes for ComfyUI.
"""

import os
import folder_paths


class SDScriptsDatasetConfig:
    """Dataset config builder for sd-scripts --dataset_config."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to the training image directory."
                }),
                "resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Training resolution (square)."
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Training batch size."
                }),
                "num_repeats": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Repeats per image in the subset."
                }),
            },
            "optional": {
                "caption_extension": ("STRING", {
                    "default": ".txt",
                    "tooltip": "Caption file extension."
                }),
                "enable_bucket": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable aspect ratio bucketing."
                }),
                "shuffle_caption": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Shuffle caption tags."
                }),
                "keep_tokens": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Keep this many tokens before shuffling."
                }),
                "min_bucket_reso": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Min bucket resolution (0 = auto)."
                }),
                "max_bucket_reso": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Max bucket resolution (0 = auto)."
                }),
                "bucket_reso_steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Bucket resolution step (0 = default)."
                }),
                "bucket_no_upscale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable upscaling when bucketing."
                }),
                "caption_dropout_rate": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Caption dropout rate (0 = off)."
                }),
                "caption_dropout_every_n_epochs": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Caption dropout every N epochs (0 = off)."
                }),
                "caption_tag_dropout_rate": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Caption tag dropout rate (0 = off)."
                }),
            }
        }

    RETURN_TYPES = ("DATASET_CONFIG",)
    RETURN_NAMES = ("dataset_config",)
    FUNCTION = "create_config"
    CATEGORY = "sd-scripts/training"

    def create_config(
        self,
        image_dir,
        resolution,
        batch_size,
        num_repeats,
        caption_extension=".txt",
        enable_bucket=True,
        shuffle_caption=True,
        keep_tokens=1,
        min_bucket_reso=0,
        max_bucket_reso=0,
        bucket_reso_steps=0,
        bucket_no_upscale=False,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
    ):
        config = {
            "image_dir": image_dir,
            "resolution": resolution,
            "batch_size": batch_size,
            "num_repeats": num_repeats,
            "caption_extension": caption_extension,
            "enable_bucket": enable_bucket,
            "shuffle_caption": shuffle_caption,
            "keep_tokens": keep_tokens,
        }

        if min_bucket_reso > 0:
            config["min_bucket_reso"] = min_bucket_reso
        if max_bucket_reso > 0:
            config["max_bucket_reso"] = max_bucket_reso
        if bucket_reso_steps > 0:
            config["bucket_reso_steps"] = bucket_reso_steps
        if bucket_no_upscale:
            config["bucket_no_upscale"] = True
        if caption_dropout_rate > 0:
            config["caption_dropout_rate"] = caption_dropout_rate
        if caption_dropout_every_n_epochs > 0:
            config["caption_dropout_every_n_epochs"] = caption_dropout_every_n_epochs
        if caption_tag_dropout_rate > 0:
            config["caption_tag_dropout_rate"] = caption_tag_dropout_rate

        return (config,)


class SDScriptsTrainParams:
    """Training parameter builder for sdxl_train_network.py."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to the SDXL base model (.safetensors or Diffusers directory)."
                }),
                "network_dim": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "LoRA rank (dim)."
                }),
                "network_alpha": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "LoRA alpha."
                }),
                "learning_rate": ("FLOAT", {
                    "default": 1e-4,
                    "min": 1e-7,
                    "max": 1e-2,
                    "step": 1e-5,
                    "tooltip": "Base learning rate."
                }),
                "max_train_epochs": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Max training epochs (ignored if max_train_steps > 0)."
                }),
            },
            "optional": {
                "unet_lr": ("FLOAT", {
                    "default": 1e-4,
                    "min": 0.0,
                    "max": 1e-2,
                    "step": 1e-5,
                    "tooltip": "UNet learning rate (0 = use base)."
                }),
                "text_encoder_lr": ("FLOAT", {
                    "default": 1e-5,
                    "min": 0.0,
                    "max": 1e-3,
                    "step": 1e-6,
                    "tooltip": "Text encoder base learning rate (0 = use base)."
                }),
                "text_encoder_lr1": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1e-3,
                    "step": 1e-6,
                    "tooltip": "Text Encoder 1 learning rate (0 = use text_encoder_lr)."
                }),
                "text_encoder_lr2": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1e-3,
                    "step": 1e-6,
                    "tooltip": "Text Encoder 2 learning rate (0 = use text_encoder_lr)."
                }),
                "optimizer_type": (["AdamW8bit", "AdamW", "Adafactor", "Lion8bit"], {
                    "default": "AdamW8bit",
                    "tooltip": "Optimizer type."
                }),
                "lr_scheduler": (["constant", "cosine", "cosine_with_restarts", "polynomial"], {
                    "default": "constant",
                    "tooltip": "Learning rate scheduler."
                }),
                "mixed_precision": (["bf16", "fp16", "no"], {
                    "default": "bf16",
                    "tooltip": "Mixed precision mode."
                }),
                "gradient_checkpointing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable gradient checkpointing."
                }),
                "cache_latents": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache latents to reduce VRAM."
                }),
                "cache_latents_to_disk": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache latents to disk when enabled."
                }),
                "cache_text_encoder_outputs": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache text encoder outputs (forces UNet-only training)."
                }),
                "cache_text_encoder_outputs_to_disk": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache text encoder outputs to disk when enabled."
                }),
                "train_unet_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Train UNet only (overridden to true when caching text encoder outputs)."
                }),
                "no_half_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Run VAE in fp32 when using fp16/bf16."
                }),
                "save_every_n_epochs": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Save every N epochs."
                }),
                "max_train_steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000000,
                    "step": 1,
                    "tooltip": "Max training steps (0 = use epochs)."
                }),
                "gradient_accumulation_steps": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Gradient accumulation steps."
                }),
                "lr_warmup_steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "LR warmup steps (0 = off)."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Random seed."
                }),
            }
        }

    RETURN_TYPES = ("TRAIN_PARAMS",)
    RETURN_NAMES = ("train_params",)
    FUNCTION = "create_params"
    CATEGORY = "sd-scripts/training"

    def create_params(
        self,
        base_model_path,
        network_dim,
        network_alpha,
        learning_rate,
        max_train_epochs,
        unet_lr=1e-4,
        text_encoder_lr=1e-5,
        text_encoder_lr1=0.0,
        text_encoder_lr2=0.0,
        optimizer_type="AdamW8bit",
        lr_scheduler="constant",
        mixed_precision="bf16",
        gradient_checkpointing=True,
        cache_latents=True,
        cache_latents_to_disk=True,
        cache_text_encoder_outputs=True,
        cache_text_encoder_outputs_to_disk=True,
        train_unet_only=False,
        no_half_vae=False,
        save_every_n_epochs=1,
        max_train_steps=0,
        gradient_accumulation_steps=1,
        lr_warmup_steps=0,
        seed=42,
    ):
        params = {
            "base_model_path": base_model_path,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "learning_rate": learning_rate,
            "max_train_epochs": max_train_epochs,
            "unet_lr": unet_lr,
            "text_encoder_lr": text_encoder_lr,
            "text_encoder_lr1": text_encoder_lr1,
            "text_encoder_lr2": text_encoder_lr2,
            "optimizer_type": optimizer_type,
            "lr_scheduler": lr_scheduler,
            "mixed_precision": mixed_precision,
            "gradient_checkpointing": gradient_checkpointing,
            "cache_latents": cache_latents,
            "cache_latents_to_disk": cache_latents_to_disk,
            "cache_text_encoder_outputs": cache_text_encoder_outputs,
            "cache_text_encoder_outputs_to_disk": cache_text_encoder_outputs_to_disk,
            "train_unet_only": train_unet_only,
            "no_half_vae": no_half_vae,
            "save_every_n_epochs": save_every_n_epochs,
            "max_train_steps": max_train_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lr_warmup_steps": lr_warmup_steps,
            "seed": seed,
        }
        return (params,)


class SDScriptsLoRATrain:
    """LoRA training node for sd-scripts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_config": ("DATASET_CONFIG",),
                "train_params": ("TRAIN_PARAMS",),
                "output_name": ("STRING", {
                    "default": "my_lora",
                    "multiline": False,
                    "tooltip": "Output LoRA name."
                }),
                "sd_scripts_path": ("STRING", {
                    "default": "c:\\sd-scripts",
                    "tooltip": "Path to the sd-scripts repository."
                }),
            },
            "optional": {
                "output_folder": ("STRING", {
                    "default": "loras",
                    "tooltip": "Subfolder under models/loras (empty or 'loras' = base)."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    FUNCTION = "train_lora"
    CATEGORY = "sd-scripts/training"
    OUTPUT_NODE = True

    def _resolve_output_dir(self, output_folder: str) -> str:
        models_dir = folder_paths.models_dir
        loras_dir = os.path.join(models_dir, "loras")
        os.makedirs(loras_dir, exist_ok=True)

        output_subdir = (output_folder or "").strip()
        output_subdir = output_subdir.replace("/", os.sep).replace("\\", os.sep)
        if output_subdir:
            drive, _ = os.path.splitdrive(output_subdir)
            if drive or os.path.isabs(output_subdir) or output_subdir.startswith("\\"):
                raise ValueError("output_folder must be relative to models/loras")
            output_subdir = output_subdir.strip(os.sep)

        if not output_subdir or output_subdir.lower() == "loras":
            output_dir = loras_dir
        elif output_subdir.lower().startswith("loras" + os.sep):
            output_dir = os.path.join(models_dir, output_subdir)
        else:
            output_dir = os.path.join(loras_dir, output_subdir)

        loras_norm = os.path.normcase(os.path.abspath(loras_dir))
        output_norm = os.path.normcase(os.path.abspath(output_dir))
        if output_norm != loras_norm and not output_norm.startswith(loras_norm + os.sep):
            raise ValueError("output_folder must stay within models/loras")

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def train_lora(self, dataset_config, train_params, output_name,
                   output_folder="loras", sd_scripts_path="c:\\sd-scripts"):
        from .trainer import SDScriptsTrainer

        trainer = SDScriptsTrainer(sd_scripts_path)

        output_dir = self._resolve_output_dir(output_folder)

        lora_path = trainer.train(
            dataset_config=dataset_config,
            train_params=train_params,
            output_name=output_name,
            output_dir=output_dir,
        )

        return (lora_path,)

