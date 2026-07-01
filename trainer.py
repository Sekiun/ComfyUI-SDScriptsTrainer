"""
sd-scripts subprocess launcher for LoRA training.
"""

import os
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)


def normalize_filesystem_path(value):
    if value is None:
        return value

    path = str(value).strip()
    if not path:
        return path

    path = os.path.expandvars(os.path.expanduser(path))
    if os.name != "nt":
        path = path.replace("\\", "/")
        home_dir = os.path.expanduser("~")
        if path == "/user" or path.startswith("/user/"):
            path = home_dir + path[len("/user"):]
        elif path == "user" or path.startswith("user/"):
            remainder = path[len("user/"):] if path != "user" else ""
            path = os.path.join(home_dir, remainder)

    return os.path.abspath(path)


class SDScriptsTrainer:
    """Run sd-scripts LoRA training via accelerate."""

    def __init__(self, sd_scripts_path: str):
        self.sd_scripts_path = normalize_filesystem_path(sd_scripts_path)
        self.venv_python, self.accelerate_path = self._resolve_venv_paths()
        self.train_scripts = {
            "sdxl": os.path.join(self.sd_scripts_path, "sdxl_train_network.py"),
            "anima": os.path.join(self.sd_scripts_path, "anima_train_network.py"),
        }
        self.train_script = self.train_scripts["sdxl"]
        self._supports_split_text_encoder_lr = self._detect_text_encoder_lr_split(self.train_script)
        self._supports_qwen_image_vae_2d = self._script_contains(self.train_scripts["anima"], "qwen_image_vae_2d")

        if not os.path.exists(self.venv_python):
            raise FileNotFoundError(
                "Python not found in venv. Tried: "
                f"{os.path.join(self.sd_scripts_path, 'venv', 'Scripts', 'python.exe')} and "
                f"{os.path.join(self.sd_scripts_path, 'venv', 'bin', 'python')}"
            )
        if not os.path.exists(self.train_scripts["sdxl"]) and not os.path.exists(self.train_scripts["anima"]):
            raise FileNotFoundError(
                "Train script not found. Tried: "
                f"{self.train_scripts['sdxl']} and {self.train_scripts['anima']}"
            )

    def _resolve_venv_paths(self):
        windows_python = os.path.join(self.sd_scripts_path, "venv", "Scripts", "python.exe")
        windows_accelerate = os.path.join(self.sd_scripts_path, "venv", "Scripts", "accelerate.exe")
        linux_python = os.path.join(self.sd_scripts_path, "venv", "bin", "python")
        linux_accelerate = os.path.join(self.sd_scripts_path, "venv", "bin", "accelerate")

        if os.path.exists(windows_python):
            return windows_python, windows_accelerate
        if os.path.exists(linux_python):
            return linux_python, linux_accelerate
        return windows_python, windows_accelerate

    def _toml_str(self, value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', "\\\"")
        return f"\"{escaped}\""

    def _script_contains(self, script_path: str, needle: str) -> bool:
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                contents = f.read()
            return needle in contents
        except OSError:
            return False

    def _detect_text_encoder_lr_split(self, script_path: str) -> bool:
        return self._script_contains(script_path, "text_encoder_lr1") or self._script_contains(script_path, "--text_encoder_lr1")

    def _normalize_model_type(self, model_type: str) -> str:
        normalized = (model_type or "sdxl").strip().lower()
        if normalized not in self.train_scripts:
            raise ValueError(f"Unsupported model_type: {model_type}")
        return normalized

    def _train_script_for_model(self, model_type: str) -> str:
        train_script = self.train_scripts[self._normalize_model_type(model_type)]
        if not os.path.exists(train_script):
            raise FileNotFoundError(f"Train script not found: {train_script}")
        return train_script

    def _create_dataset_toml(self, dataset_config: dict, temp_dir: str) -> str:
        toml_path = os.path.join(temp_dir, "dataset_config.toml")

        image_dir = normalize_filesystem_path(dataset_config["image_dir"])
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

    def _base_command(self, train_script: str) -> list:
        if os.path.exists(self.accelerate_path):
            return [
                self.accelerate_path,
                "launch",
                "--num_cpu_threads_per_process", "1",
                train_script,
            ]
        return [
            self.venv_python,
            "-m", "accelerate",
            "launch",
            "--num_cpu_threads_per_process", "1",
            train_script,
        ]

    def _append_optional_path(self, cmd: list, option: str, value):
        if value is not None and str(value).strip():
            cmd.append(f"{option}={normalize_filesystem_path(value)}")

    def _append_positive_int(self, cmd: list, option: str, value):
        int_value = int(value or 0)
        if int_value > 0:
            cmd.append(f"{option}={int_value}")

    def _append_float_if_set(self, cmd: list, option: str, value):
        if value is None:
            return
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            return
        if float_value < 0:
            return
        cmd.append(f"{option}={float_value}")

    def _append_network_args(self, cmd: list, train_params: dict):
        network_args = []
        if train_params.get("train_llm_adapter", False):
            network_args.append("train_llm_adapter=True")
        for key in ("network_reg_dims", "network_reg_lrs", "include_patterns", "exclude_patterns"):
            value = train_params.get(key)
            if value is not None and str(value).strip():
                network_args.append(f"{key}={str(value).strip()}")
        if network_args:
            cmd.append("--network_args")
            cmd.extend(network_args)

    def _append_common_options(self, cmd: list, train_params: dict):
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

    def _append_cache_options(self, cmd: list, dataset_config: dict, train_params: dict) -> bool:
        cache_latents = bool(train_params.get("cache_latents", True))
        if cache_latents:
            cmd.append("--cache_latents")
            if train_params.get("cache_latents_to_disk", True):
                cmd.append("--cache_latents_to_disk")

        cache_te = bool(train_params.get("cache_text_encoder_outputs", True))
        if cache_te and dataset_config.get("shuffle_caption", True):
            cache_te = False
            logger.warning("Disabled cache_text_encoder_outputs because shuffle_caption is enabled.")
        if cache_te and float(dataset_config.get("caption_dropout_rate", 0) or 0) > 0:
            cache_te = False
            logger.warning("Disabled cache_text_encoder_outputs because caption_dropout_rate is enabled.")
        if cache_te and float(dataset_config.get("caption_tag_dropout_rate", 0) or 0) > 0:
            cache_te = False
            logger.warning("Disabled cache_text_encoder_outputs because caption_tag_dropout_rate is enabled.")
        if cache_te and int(dataset_config.get("caption_dropout_every_n_epochs", 0) or 0) > 0:
            cache_te = False
            logger.warning("Disabled cache_text_encoder_outputs because caption_dropout_every_n_epochs is enabled.")
        if cache_te:
            cmd.append("--cache_text_encoder_outputs")
            if train_params.get("cache_text_encoder_outputs_to_disk", True):
                cmd.append("--cache_text_encoder_outputs_to_disk")
        return cache_te

    def _build_command(self, dataset_toml: str, dataset_config: dict, train_params: dict,
                       output_name: str, output_dir: str) -> list:
        model_type = self._normalize_model_type(train_params.get("model_type", "sdxl"))
        train_script = self._train_script_for_model(model_type)
        cmd = self._base_command(train_script)

        learning_rate = float(train_params["learning_rate"])
        base_model_path = normalize_filesystem_path(train_params["base_model_path"])

        cmd.extend([
            f"--pretrained_model_name_or_path={base_model_path}",
            f"--dataset_config={dataset_toml}",
            f"--output_dir={output_dir}",
            f"--output_name={output_name}",
            "--save_model_as=safetensors",
            f"--network_dim={train_params['network_dim']}",
            f"--network_alpha={train_params['network_alpha']}",
            f"--learning_rate={learning_rate}",
            f"--optimizer_type={train_params.get('optimizer_type', 'AdamW8bit')}",
            f"--lr_scheduler={train_params.get('lr_scheduler', 'constant')}",
        ])

        if model_type == "anima":
            qwen3_path = normalize_filesystem_path(train_params.get("qwen3_path", ""))
            vae_path = normalize_filesystem_path(train_params.get("vae_path", ""))
            if not qwen3_path:
                raise ValueError("qwen3_path is required when model_type is anima")
            if not vae_path:
                raise ValueError("vae_path is required when model_type is anima")

            cmd.extend([
                "--network_module=networks.lora_anima",
                f"--qwen3={qwen3_path}",
                f"--vae={vae_path}",
                f"--timestep_sampling={train_params.get('timestep_sampling', 'sigmoid')}",
                f"--discrete_flow_shift={float(train_params.get('discrete_flow_shift', 1.0) or 1.0)}",
                f"--sigmoid_scale={float(train_params.get('sigmoid_scale', 1.0) or 1.0)}",
                f"--qwen3_max_token_length={int(train_params.get('qwen3_max_token_length', 512) or 512)}",
                f"--t5_max_token_length={int(train_params.get('t5_max_token_length', 512) or 512)}",
            ])
            self._append_optional_path(cmd, "--llm_adapter_path", train_params.get("llm_adapter_path"))
            self._append_optional_path(cmd, "--t5_tokenizer_path", train_params.get("t5_tokenizer_path"))
            self._append_optional_path(cmd, "--attn_mode", train_params.get("attn_mode"))
            self._append_positive_int(cmd, "--blocks_to_swap", train_params.get("blocks_to_swap"))
            self._append_positive_int(cmd, "--vae_chunk_size", train_params.get("vae_chunk_size"))
            self._append_float_if_set(cmd, "--self_attn_lr", train_params.get("self_attn_lr"))
            self._append_float_if_set(cmd, "--cross_attn_lr", train_params.get("cross_attn_lr"))
            self._append_float_if_set(cmd, "--mlp_lr", train_params.get("mlp_lr"))
            self._append_float_if_set(cmd, "--mod_lr", train_params.get("mod_lr"))
            self._append_float_if_set(cmd, "--llm_adapter_lr", train_params.get("llm_adapter_lr"))
            if train_params.get("split_attn", False):
                cmd.append("--split_attn")
            if train_params.get("vae_disable_cache", False):
                cmd.append("--vae_disable_cache")
            if train_params.get("qwen_image_vae_2d", False) and self._supports_qwen_image_vae_2d:
                cmd.append("--qwen_image_vae_2d")
            self._append_network_args(cmd, train_params)
        else:
            unet_lr = self._resolve_lr(train_params.get("unet_lr"), learning_rate)
            text_lr_base = self._resolve_lr(train_params.get("text_encoder_lr"), learning_rate)
            text_lr1 = self._resolve_lr(train_params.get("text_encoder_lr1"), text_lr_base)
            text_lr2 = self._resolve_lr(train_params.get("text_encoder_lr2"), text_lr_base)

            cmd.extend([
                "--network_module=networks.lora",
                f"--unet_lr={unet_lr}",
            ])
            if self._supports_split_text_encoder_lr:
                cmd.extend([
                    f"--text_encoder_lr1={text_lr1}",
                    f"--text_encoder_lr2={text_lr2}",
                ])
            else:
                cmd.extend([
                    "--text_encoder_lr",
                    str(text_lr1),
                    str(text_lr2),
                ])

        self._append_common_options(cmd, train_params)
        cache_te = self._append_cache_options(cmd, dataset_config, train_params)

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
            cmd = self._build_command(dataset_toml, dataset_config, train_params, output_name, output_dir)

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
