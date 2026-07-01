# ComfyUI-SDScriptsTrainer

ComfyUI から `sd-scripts` を使って SDXL LoRA 学習を実行するカスタムノードです。

## 前提
- `sd-scripts` 本体があり、`sd-scripts\venv` が作成済み。
- GPU で学習する場合は CUDA 対応の PyTorch が必要。

## GPU 設定（sd-scripts venv）
インストール状況の確認:
```powershell
C:\sd-scripts\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

CUDA 12.1 版の例:
```powershell
C:\sd-scripts\venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## よくあるエラー
- `ModuleNotFoundError: torchvision`
  - sd-scripts の venv に入れます:
    ```powershell
    C:\sd-scripts\venv\Scripts\python.exe -m pip install torchvision
    ```

- `error: unrecognized arguments: --text_encoder_lr1/--text_encoder_lr2`
  - `sd-scripts` 側が `--text_encoder_lr` を想定している場合に発生します。
  - 本ノードは自動判定で `--text_encoder_lr <lr1> <lr2>` に切り替えます。
  - それでも出る場合は `sd-scripts` を更新してください。

- `AssertionError: when caching Text Encoder output ... shuffle_caption ... caption_dropout ...`
  - Text Encoder 出力キャッシュと `shuffle_caption`/caption dropout は併用できません。
  - 本ノードはそれらが有効なとき TE キャッシュを自動で無効化します（警告を表示）。
  - TE キャッシュを使いたい場合は `shuffle_caption = false`、caption dropout を 0 にしてください。

- `accelerate launch` の警告（デフォルト値使用）
  - 必要なら sd-scripts venv で `accelerate config` を実行してください。

## 謝辞
- 本ノードは kohya-ss 氏の `sd-scripts` を利用しています: https://github.com/kohya-ss/sd-scripts
- LoRA 実装は cloneofsimo のリポジトリに基づきます: https://github.com/cloneofsimo/lora
- Conv2d 3x3 への LoRA 拡張は KohakuBlueleaf の LoCon によるものです: https://github.com/KohakuBlueleaf/LoCon

## 反映
Python パッケージやカスタムノードを変更したら ComfyUI を再起動してください。

## Anima LoRA
Set `model_type` in `SDScriptsTrainParams` to `anima` to run `anima_train_network.py` with `networks.lora_anima`.

Required inputs:
- `base_model_path`: Anima DiT `.safetensors` file
- `qwen3_path`: Qwen3-0.6B text encoder file or directory
- `vae_path`: Qwen-Image VAE `.safetensors` or `.pth` file

Main Anima options:
- `timestep_sampling`: `sigmoid`, `sigma`, `uniform`, `shift`, or `flux_shift`
- `discrete_flow_shift`, `sigmoid_scale`
- `blocks_to_swap`, `vae_chunk_size`, `vae_disable_cache`
- `train_llm_adapter`, `network_reg_dims`, `network_reg_lrs`, `include_patterns`, `exclude_patterns`

On the Linux server, the default `sd_scripts_path` is `~/sd-scripts`.

