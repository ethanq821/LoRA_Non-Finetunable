#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行方式：
    python data4test.py
"""

import os
import pprint

import torch
from accelerate import Accelerator
from transformers import CLIPTokenizer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig
from PIL import Image

from data import (
    prepare_dataset,
    DreamBoothDataset,
    dreambooth_collate_fn,
    set_seed,
)

def load_cfg(cfg_root: str) -> DictConfig:
    """
    使用 Hydra compose 加载 cfg_root/config.yaml，返回 DictConfig。
    """
    cfg_root = os.path.abspath(cfg_root)
    cfg_file = os.path.join(cfg_root, "config.yaml")
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"未找到配置文件: {cfg_file}")

    # 使用 Hydra 处理 defaults 等引用
    with initialize_config_dir(config_dir=cfg_root, job_name="data4test", version_base=None):
        cfg = compose(config_name="config")

    return cfg  # DictConfig 支持点语法访问，如 cfg.datasets.instance_prompt


def make_accelerator() -> Accelerator:
    return Accelerator()


def inspect_dataloader(dataloader, label: str):
    """
    取出一个 batch，并通过 pdb 查看张量与 token。
    """
    print(f"\n===== {label}: 取首个批次并进入 pdb 调试 =====")
    batch = next(iter(dataloader))
    print("Batch keys:")
    pprint.pp(batch.keys())

    import pdb
    pdb.set_trace()


def main():
    # 1. 加载配置
    cfg_root = "/home/dev/zbq/lora_nft_Oct/lora_nonfinetunable/cfg"
    cfg = load_cfg(cfg_root)

    dataset_root = "/data1/dif/non-finetune-lora/test_case/target/0"
    cache_dir = "not used"
    use_dream_booth = True

    # 3. 初始化环境
    accelerator = make_accelerator()
    tokenizer = CLIPTokenizer.from_pretrained("/data1/dif/sd1-5/tokenizer")
    set_seed(cfg.seed)

    # 4. 根据配置执行常规/NFT训练数据集测试
    print("\n========== 调用 prepare_dataset ==========")
    train_dataset, train_dataloader, nft_dataset, nft_dataloader = prepare_dataset(
        cfg=cfg,
        dataset_name=dataset_root,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        accelerator=accelerator,
        use_dream_booth=use_dream_booth,
        class_data_dir=None,
        pre_computed_encoder_hidden_states=None,
        pre_computed_class_prompt_encoder_hidden_states=None,
    )

    # 5. 打印并检查数据集对象类型
    print("train_dataset 类型:", type(train_dataset))
    print("train_dataloader 类型:", type(train_dataloader))
    print("nft_dataset 类型:", type(nft_dataset))
    print("nft_dataloader 类型:", type(nft_dataloader))

    # # 6. 使用 pdb 观察 batch
    # inspect_dataloader(train_dataloader, "Train DataLoader")
    # if nft_dataloader is not None:
    #     inspect_dataloader(nft_dataloader, "NFT DataLoader")

    import os
    from PIL import Image

    output_dir = "test_dataset"
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(1): 
        batch = next(iter(train_dataloader))
        print(batch.keys())
        img = batch["pixel_values"][0]        # 取第 0 张
        img = (img / 2 + 0.5).clamp(0, 1)
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(img_np).save(os.path.join(output_dir, f"sample_train_{idx}.png"))
        ids = batch["input_ids"][0]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"sample_train_{idx}.png -> {prompt}")
        
        batch = next(iter(nft_dataloader))
        print(batch.keys())
        img = batch["pixel_values"][0]        # 取第 0 张
        img = (img / 2 + 0.5).clamp(0, 1)
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(img_np).save(os.path.join(output_dir, f"sample_nft_{idx}.png"))
        ids = batch["input_ids"][0]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"sample_nft_{idx}.png -> {prompt}")


if __name__ == "__main__":
    main()
