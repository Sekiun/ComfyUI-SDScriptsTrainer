# 古いプリセット
今更使うものでもない古くなったプリセットです。このプリセットはサポート対象外となります。

・キャラLoRA_4096steps_IN04-07_OUT03-06のみ.xmlora ※SD1.5
人物の特徴に影響しない層を無効にしたものです。画風の変化を抑えつつ、ファイルサイズを減らします。
supermergerのドキュメントを参考にしました: https://github.com/hako-mikan/sd-webui-supermerger/blob/main/elemental_ja.md

・画風.xmlora ※SD1.5
画風向けです。総ステップ数(batch1換算)は4000-8000stepsを推奨します。

・SDXL汎用プリセット.xmlora
Stable Diffusion XL向けの設定です。
次元数を16、アルファを4にしてもいいかもしれません。
OptimizerはAdamW8bitがうまく学習できない可能性があるため、Lionにしました。

・SDXL(PonyV6XL).xmlora
ほとんどの用途に対応するプリセットです。 
詳細設定->パスでVAEに https://huggingface.co/madebyollin/sdxl-vae-fp16-fix で配布されているsdxl.vae.safetensorsを選択してください。
キャラの学習は2500-3500steps(batch1換算)を推奨します。

・Animagine汎用プリセット.xmlora
ほとんどの用途に対応するプリセットです。
キャラの学習は2000-3000steps(batch1換算)を推奨します。
学習元モデルは以下のURLにあるものを使用すると意図しない画風の変化を軽減します。
https://civitai.com/models/405165/genimagine-xlfor-lora-training?modelVersionId=451728

・Animagine汎用プリセット_dora.xmlora
Weight-Decomposed Low-Rank Adaptation(DoRA)を使用することで若干の精度改善および大幅な安定性の向上が期待できます。
2000-3000ステップ(batch1換算)で十分な結果を得られます。
学習時の計算速度が20%以上遅くなります。
現在、成果物は1111WebUIの1.9.0以降またはComfyUIで使用できます。それ以外では反映されないかエラーになります。
注意:Ponyは発散して学習できません。

・SDXL汎用プリセット_高速学習(LoRA+).xmlora
LoRA+を使用することで通常のLoRAの半分未満のステップ数で学習できるものになります。しかし、学習内容によっては従来のLoRAより品質が低下することがあるようです。
単一キャラの学習であれば、総ステップ数(batch1換算)は800～1100に設定してください。
上記のステップ数であれば、RTX 3060tiや2080相当のGPUで20-30分で学習できます！多分4070Tiで13分、4090で9分でできる？
VRAMが12GB以上ある場合は「詳細設定」->「パフォーマンス」 にある「モデルをfp8で読み込む」をオフにできます。

・SDXLキャラ_高速学習(LoRA+).xmlora
Transformerの多い層のみにしたものです。画風の変化を抑えつつ、ファイルサイズを減らします。
単一キャラの学習であれば、総ステップ数(batch1換算)は900～1300に設定してください。