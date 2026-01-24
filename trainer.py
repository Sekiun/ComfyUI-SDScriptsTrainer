"""
sd-scripts subprocess launcher for SDXL LoRA training.
"""

import os
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)


class SDScriptsTrainer:
    """Run sd-scripts SDXL LoRA training via accelerate."""

    def __init__(self, sd_scripts_path: str):
        self.sd_scripts_path = os.path.abspath(os.path.expandvars(os.path.expanduser(sd_scripts_path)))
        self.venv_python = os.path.join(self.sd_scripts_path, "venv", "Scripts", "python.exe")
        self.accelerate_path = os.path.join(self.sd_scripts_path, "venv", "Scripts", "accelerate.exe")
        self.train_script = os.path.join(self.sd_scripts_path, "sdxl_train_network.py")

        if not os.path.exists(self.venv_python):
            raise FileNotFoundError(f"Python not found: {self.venv_python}")
        if not os.path.exists(self.train_script):
            raise FileNotFoundError(f"Train script not found: {self.train_script}")

    def _toml_str(self, value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', "\\\"")
        return f"\"{escaped}\""

    def _create_dataset_toml(self, dataset_config: dict, temp_dir: str) -> str:
        toml_path = os.path.join(temp_dir, "dataset_config.toml")

        image_dir = dataset_config["image_dir"]
        lines = [
            "[general]",
            f"shuffle_caption = {str(dataset_config.get('shuffle_caption', True)).lower()}",
            f"caption_extension = {self._toml_str(dataset_config.get('caption_extension', '.txt'))}",
            f"keep_tokens = {int(dataset_config.get('keep_tokens', 1))}",
        ]

        caption_dropout_rate = float(dataset_config.get("caption_dropout_rate", 0) or 0)
        if caption_dropout_rate > 0:
            lines.append(f"caption_dropout_rate = {caption_dropout_rate}")

        caption_dropout_every_n_epochs = int(dataset_config.get("caption_dropout_every_n_epochs", 0) or 0)
        if caption_dropout_every_n_epochs > 0:
            lines.append(f"caption_dropout_every_n_epochs = {caption_dropout_every_n_epochs}")

        caption_tag_dropout_rate = float(dataset_config.get("caption_tag_dropout_rate", 0) or 0)
        if caption_tag_dropout_rate > 0:
            lines.append(f"caption_tag_dropout_rate = {caption_tag_dropout_rate}")

        lines += [
            "",
            "[[datasets]]",
            f"resolution = {int(dataset_config['resolution'])}",
            f"batch_size = {int(dataset_config['batch_size'])}",
            f"enable_bucket = {str(dataset_config.get('enable_bucket', True)).lower()}",
        ]

        if dataset_config.get("bucket_no_upscale"):
            lines.append("bucket_no_upscale = true")

        min_bucket_reso = int(dataset_config.get("min_bucket_reso", 0) or 0)
        if min_bucket_reso > 0:
            lines.append(f"min_bucket_reso = {min_bucket_reso}")

        max_bucket_reso = int(dataset_config.get("max_bucket_reso", 0) or 0)
        if max_bucket_reso > 0:
            lines.append(f"max_bucket_reso = {max_bucket_reso}")

        bucket_reso_steps = int(dataset_config.get("bucket_reso_steps", 0) or 0)
        if bucket_reso_steps > 0:
            lines.append(f"bucket_reso_steps = {bucket_reso_steps}")

        lines += [
            "",
            "  [[datasets.subsets]]",
            f"  image_dir = {self._toml_str(image_dir)}",
            f"  num_repeats = {int(dataset_config.get('num_repeats', 10))}",
        ]

        toml_content = "\n".join(lines) + "\n"

        with open(toml_path, "w", encoding="utf-8") as f:
            f.write(toml_content)

        logger.info("Created dataset config: %s", toml_path)
        return toml_path

    def _resolve_lr(self, value, fallback):
        if value is None:
            return fallback
        try:
            lr_value = float(value)
        except (TypeError, ValueError):
            return fallback
        if lr_value <= 0:
            return fallback
        return lr_value

    def _build_command(self, dataset_toml: str, train_params: dict,
                       output_name: str, output_dir: str) -> list:
        if os.path.exists(self.accelerate_path):
            cmd = [
                self.accelerate_path,
                "launch",
                "--num_cpu_threads_per_process", "1",
                self.train_script,
            ]
        else:
            cmd = [
                self.venv_python,
                "-m", "accelerate",
                "launch",
                "--num_cpu_threads_per_process", "1",
                self.train_script,
            ]

        learning_rate = float(train_params["learning_rate"])
        unet_lr = self._resolve_lr(train_params.get("unet_lr"), learning_rate)
        text_lr_base = self._resolve_lr(train_params.get("text_encoder_lr"), learning_rate)
        text_lr1 = self._resolve_lr(train_params.get("text_encoder_lr1"), text_lr_base)
        text_lr2 = self._resolve_lr(train_params.get("text_encoder_lr2"), text_lr_base)

        cmd.extend([
            f"--pretrained_model_name_or_path={train_params['base_model_path']}",
            f"--dataset_config={dataset_toml}",
            f"--output_dir={output_dir}",
            f"--output_name={output_name}",
            "--save_model_as=safetensors",
            "--network_module=networks.lora",
            f"--network_dim={train_params['network_dim']}",
            f"--network_alpha={train_params['network_alpha']}",
            f"--learning_rate={learning_rate}",
            f"--unet_lr={unet_lr}",
            f"--text_encoder_lr1={text_lr1}",
            f"--text_encoder_lr2={text_lr2}",
            f"--optimizer_type={train_params.get('optimizer_type', 'AdamW8bit')}",
            f"--lr_scheduler={train_params.get('lr_scheduler', 'constant')}",
        ])

        max_train_steps = int(train_params.get("max_train_steps", 0) or 0)
        if max_train_steps > 0:
            cmd.append(f"--max_train_steps={max_train_steps}")
        else:
            cmd.append(f"--max_train_epochs={train_params['max_train_epochs']}")

        save_every_n_epochs = int(train_params.get("save_every_n_epochs", 1) or 0)
        if save_every_n_epochs > 0:
            cmd.append(f"--save_every_n_epochs={save_every_n_epochs}")

        mixed_precision = train_params.get("mixed_precision", "bf16")
        if mixed_precision:
            cmd.append(f"--mixed_precision={mixed_precision}")

        seed = train_params.get("seed")
        if seed is not None:
            cmd.append(f"--seed={seed}")

        grad_accum = int(train_params.get("gradient_accumulation_steps", 1) or 1)
        if grad_accum > 1:
            cmd.append(f"--gradient_accumulation_steps={grad_accum}")

        lr_warmup_steps = int(train_params.get("lr_warmup_steps", 0) or 0)
        if lr_warmup_steps > 0:
            cmd.append(f"--lr_warmup_steps={lr_warmup_steps}")

        if train_params.get("gradient_checkpointing", True):
            cmd.append("--gradient_checkpointing")

        cache_latents = bool(train_params.get("cache_latents", True))
        if cache_latents:
            cmd.append("--cache_latents")
            if train_params.get("cache_latents_to_disk", True):
                cmd.append("--cache_latents_to_disk")

        cache_te = bool(train_params.get("cache_text_encoder_outputs", True))
        if cache_te:
            cmd.append("--cache_text_encoder_outputs")
            if train_params.get("cache_text_encoder_outputs_to_disk", True):
                cmd.append("--cache_text_encoder_outputs_to_disk")

        train_unet_only = bool(train_params.get("train_unet_only", False))
        if cache_te:
            train_unet_only = True
        if train_unet_only:
            cmd.append("--network_train_unet_only")

        if train_params.get("no_half_vae", False):
            cmd.append("--no_half_vae")

        return cmd

    def train(self, dataset_config: dict, train_params: dict,
              output_name: str, output_dir: str) -> str:
        """
        Run LoRA training.

        Returns:
            str: Path to the generated LoRA file.
        """
        temp_dir = tempfile.mkdtemp(prefix="sdscripts_")

        try:
            dataset_toml = self._create_dataset_toml(dataset_config, temp_dir)
            cmd = self._build_command(dataset_toml, train_params, output_name, output_dir)

            logger.info("Executing command: %s", " ".join(cmd))
            print("[SD-Scripts] Starting LoRA training...")
            print(f"[SD-Scripts] Output: {output_dir}/{output_name}.safetensors")

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            process = subprocess.Popen(
                cmd,
                cwd=self.sd_scripts_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            for line in process.stdout:
                print(f"[SD-Scripts] {line.rstrip()}")

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"Training failed with return code: {process.returncode}")

            lora_path = os.path.join(output_dir, f"{output_name}.safetensors")

            if not os.path.exists(lora_path):
                for f in os.listdir(output_dir):
                    if f.startswith(output_name) and f.endswith(".safetensors"):
                        lora_path = os.path.join(output_dir, f)
                        break

            if os.path.exists(lora_path):
                print(f"[SD-Scripts] Training completed! LoRA saved to: {lora_path}")
                return lora_path

            raise FileNotFoundError(f"LoRA file not found in {output_dir}")
        finally:
            pass
