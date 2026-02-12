"""
Utilities for loading xmlora sample presets and mapping to node fields.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Iterable, Optional
from xml.etree import ElementTree as ET


_NONE_PRESET = "(none)"
_SAMPLE_PRESETS_DIR = os.path.join(
    os.path.dirname(__file__),
    "samplepresets",
    "Sample Presets",
)

_PAYLOAD_CACHE: Dict[str, Any] = {"snapshot": None, "payload": None}


def _iter_xmlora_files(base_dir: str) -> Iterable[str]:
    if not os.path.isdir(base_dir):
        return []

    results = []
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.lower().endswith(".xmlora"):
                results.append(os.path.join(root, filename))
    results.sort(key=lambda p: p.lower())
    return results


def _to_rel_name(path: str) -> str:
    rel = os.path.relpath(path, _SAMPLE_PRESETS_DIR)
    return rel.replace("\\", "/")


def _safe_text(root: ET.Element, tag: str) -> Optional[str]:
    element = root.find(tag)
    if element is None:
        return None
    text = element.text
    if text is None:
        return None
    text = text.strip()
    return text if text else None


def _to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    return None


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def _snapshot(paths: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for path in paths:
        try:
            stat = os.stat(path)
        except OSError:
            continue
        digest.update(path.encode("utf-8", errors="ignore"))
        digest.update(str(stat.st_mtime_ns).encode("ascii", errors="ignore"))
        digest.update(str(stat.st_size).encode("ascii", errors="ignore"))
    return digest.hexdigest()


def _parse_single_preset(path: str) -> Dict[str, Dict[str, Any]]:
    train: Dict[str, Any] = {}
    dataset: Dict[str, Any] = {}

    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError):
        return {"train": train, "dataset": dataset}

    root = tree.getroot()

    base_model_path = _safe_text(root, "ModelPath")
    if base_model_path:
        train["base_model_path"] = base_model_path

    network_dim = _to_int(_safe_text(root, "NetworkDim"))
    if network_dim is not None and network_dim > 0:
        train["network_dim"] = network_dim

    network_alpha = _to_int(_safe_text(root, "NetworkAlpha"))
    if network_alpha is not None and network_alpha > 0:
        train["network_alpha"] = network_alpha

    learning_rate = _to_float(_safe_text(root, "LearningRate"))
    if learning_rate is not None and learning_rate > 0:
        train["learning_rate"] = learning_rate

    max_train_epochs = _to_int(_safe_text(root, "Epochs"))
    if max_train_epochs is not None and max_train_epochs > 0:
        train["max_train_epochs"] = max_train_epochs

    unet_lr = _to_float(_safe_text(root, "UnetLR"))
    if unet_lr is not None:
        train["unet_lr"] = unet_lr if unet_lr > 0 else 0.0

    text_encoder_lr = _to_float(_safe_text(root, "TextEncoderLR"))
    if text_encoder_lr is not None:
        train["text_encoder_lr"] = text_encoder_lr if text_encoder_lr > 0 else 0.0

    optimizer_type = _safe_text(root, "OptimizerType")
    if optimizer_type:
        train["optimizer_type"] = optimizer_type

    lr_scheduler = _safe_text(root, "SchedulerType")
    if lr_scheduler:
        train["lr_scheduler"] = lr_scheduler

    mixed_precision = _safe_text(root, "mixedPrecisionType")
    if mixed_precision:
        train["mixed_precision"] = mixed_precision

    gradient_checkpointing = _to_bool(_safe_text(root, "UseGradient"))
    if gradient_checkpointing is not None:
        train["gradient_checkpointing"] = gradient_checkpointing

    cache_latents = _to_bool(_safe_text(root, "CacheLatents"))
    if cache_latents is not None:
        train["cache_latents"] = cache_latents

    cache_latents_to_disk = _to_bool(_safe_text(root, "CacheLatentsToDisk"))
    if cache_latents_to_disk is not None:
        train["cache_latents_to_disk"] = cache_latents_to_disk

    cache_text_encoder_outputs = _to_bool(_safe_text(root, "CacheTextencoder"))
    if cache_text_encoder_outputs is not None:
        train["cache_text_encoder_outputs"] = cache_text_encoder_outputs

    cache_text_encoder_outputs_to_disk = _to_bool(_safe_text(root, "CacheTextencoderToDisk"))
    if cache_text_encoder_outputs_to_disk is not None:
        train["cache_text_encoder_outputs_to_disk"] = cache_text_encoder_outputs_to_disk

    no_half_vae = _to_bool(_safe_text(root, "NoHalfVAE"))
    if no_half_vae is not None:
        train["no_half_vae"] = no_half_vae

    save_every_n_epochs = _to_int(_safe_text(root, "SaveEveryNEpochs"))
    if save_every_n_epochs is not None:
        train["save_every_n_epochs"] = max(1, save_every_n_epochs)

    gradient_accumulation_steps = _to_int(_safe_text(root, "GradAccSteps"))
    if gradient_accumulation_steps is not None and gradient_accumulation_steps > 0:
        train["gradient_accumulation_steps"] = gradient_accumulation_steps

    lr_warmup_steps = _to_int(_safe_text(root, "WarmupSteps"))
    if lr_warmup_steps is not None and lr_warmup_steps >= 0:
        train["lr_warmup_steps"] = lr_warmup_steps

    seed = _to_int(_safe_text(root, "Seed"))
    if seed is not None and seed >= 0:
        train["seed"] = seed

    resolution = _to_int(_safe_text(root, "Resolution"))
    if resolution is not None and resolution > 0:
        dataset["resolution"] = resolution

    batch_size = _to_int(_safe_text(root, "BatchSize"))
    if batch_size is not None and batch_size > 0:
        dataset["batch_size"] = batch_size

    caption_extension = _safe_text(root, "CaptionFileExtension")
    if caption_extension:
        dataset["caption_extension"] = caption_extension

    shuffle_caption = _to_bool(_safe_text(root, "ShuffleCaptions"))
    if shuffle_caption is not None:
        dataset["shuffle_caption"] = shuffle_caption

    keep_tokens = _to_int(_safe_text(root, "KeepTokenCount"))
    if keep_tokens is not None and keep_tokens >= 0:
        dataset["keep_tokens"] = keep_tokens

    min_bucket_reso = _to_int(_safe_text(root, "MinBucketResolution"))
    if min_bucket_reso is not None and min_bucket_reso >= 0:
        dataset["min_bucket_reso"] = min_bucket_reso

    max_bucket_reso = _to_int(_safe_text(root, "MaxBucketResolution"))
    if max_bucket_reso is not None and max_bucket_reso >= 0:
        dataset["max_bucket_reso"] = max_bucket_reso

    bucket_no_upscale = _to_bool(_safe_text(root, "NoBucketUpscaling"))
    if bucket_no_upscale is not None:
        dataset["bucket_no_upscale"] = bucket_no_upscale

    caption_dropout_rate = _to_float(_safe_text(root, "CaptionDropout"))
    if caption_dropout_rate is not None and caption_dropout_rate >= 0:
        dataset["caption_dropout_rate"] = caption_dropout_rate

    caption_tag_dropout_rate = _to_float(_safe_text(root, "CaptionTagDropout"))
    if caption_tag_dropout_rate is not None and caption_tag_dropout_rate >= 0:
        dataset["caption_tag_dropout_rate"] = caption_tag_dropout_rate

    return {"train": train, "dataset": dataset}


def get_preset_choices() -> list:
    paths = list(_iter_xmlora_files(_SAMPLE_PRESETS_DIR))
    names = [_to_rel_name(path) for path in paths]
    return [_NONE_PRESET] + names


def get_preset_payload() -> Dict[str, Any]:
    paths = list(_iter_xmlora_files(_SAMPLE_PRESETS_DIR))
    current_snapshot = _snapshot(paths)

    if _PAYLOAD_CACHE["snapshot"] == current_snapshot and _PAYLOAD_CACHE["payload"] is not None:
        return _PAYLOAD_CACHE["payload"]

    presets = []
    train_values: Dict[str, Dict[str, Any]] = {}
    dataset_values: Dict[str, Dict[str, Any]] = {}

    for path in paths:
        rel_name = _to_rel_name(path)
        parsed = _parse_single_preset(path)
        presets.append(rel_name)
        train_values[rel_name] = parsed["train"]
        dataset_values[rel_name] = parsed["dataset"]

    payload = {
        "none": _NONE_PRESET,
        "presets": presets,
        "train": train_values,
        "dataset": dataset_values,
    }
    _PAYLOAD_CACHE["snapshot"] = current_snapshot
    _PAYLOAD_CACHE["payload"] = payload
    return payload
