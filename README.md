# Non-Finetunable LoRA

Reference: 
[HuggingFace Finetuning Script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)

Dataset preparation directory: /data1/dif/non-finetune-lora, where the **Anime** Dataset is grabbed from the [link](https://huggingface.co/datasets/AngelBottomless/Kohaku-Delta-Alpha-Characters-Result)


# 1. Dataset Process

Related files are under /data1/dif/non-finetune-lora, check readme.md for more details.

# 2. Normal Training (Reference)

`cd lora_nonfinetunable`

`bash scripts/dream_booth_normal.sh.sh`

# 3. Protected Training

`cd lora_nonfinetunable`

`bash scripts/non_finetune.sh`


## TODO LIST
- [Done] Training Multiple LoRAs -- Hard to decouple the code, This can be done by bash command
- [Done] Config organization
- [Optional] SNR 