import os
import numpy as np
import torch
import random
import pathlib
from typing import List
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose

from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import cast_training_params

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from datasets import Dataset, DatasetDict, load_from_disk


"""
Arguments changed.
"""

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

def prepare_dataset(
    # path:str=None,
    cfg,
    dataset_name,
    cache_dir,
    tokenizer,
    accelerator,
    use_dream_booth=True,
    class_data_dir=None,
    pre_computed_encoder_hidden_states = None,
    pre_computed_class_prompt_encoder_hidden_states = None,
):  
    if not use_dream_booth:
        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_from_disk(dataset_name)
            '''
            Only for normal training!!!!!!
            '''
            dataset = DatasetDict({"train": dataset})
        else:
            raise ValueError("dataset_name is None")
            # data_files = {}
            # if cfg.datasets.train_data_dir is not None:
            #     data_files["train"] = os.path.join(cfg.datasets.train_data_dir, "**")
            # dataset = load_dataset(
            #     "imagefolder",
            #     data_files=data_files,
            #     cache_dir=cache_dir,
            # )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = ("image", "text")
        if cfg.datasets.image_column is None:
            image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            image_column = cfg.datasets.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{cfg.datasets.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if cfg.datasets.caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            caption_column = cfg.datasets.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{cfg.datasets.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Preprocessing the datasets.
        # We need to tokenize input captions and transform the images.
        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            return inputs.input_ids

        # Get the specified interpolation method from the args
        interpolation = getattr(transforms.InterpolationMode, cfg.models.image_interpolation_mode.upper(), None)

        # Raise an error if the interpolation method is invalid
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation mode {cfg.models.image_interpolation_mode}.")

        # Data preprocessing transformations
        train_transforms = transforms.Compose(
            [
                transforms.Resize(cfg.models.resolution, interpolation=interpolation),  # Use dynamic interpolation method
                transforms.CenterCrop(cfg.models.resolution) if cfg.datasets.center_crop else transforms.RandomCrop(cfg.models.resolution),
                transforms.RandomHorizontalFlip() if cfg.datasets.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)
            return examples

        with accelerator.main_process_first():
            if cfg.datasets.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=cfg.seed).select(range(cfg.datasets.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=cfg.train_batch_size,
            num_workers=cfg.dataloader_num_workers,
        )
    else:
        # Dreambooth Dataset and DataLoaders creation:
        safe_instance_dir = os.path.join(dataset_name, "safe")
        unsafe_instance_dir = os.path.join(dataset_name, "unsafe")
        unsafe_instance_prompt = cfg.datasets.instance_prompt + ", naked, nsfw, unclothed, nude"
        class_data_dir = class_data_dir

        train_dataset = DreamBoothDataset(
            instance_data_root=safe_instance_dir,
            instance_prompt=cfg.datasets.instance_prompt,
            class_data_root=class_data_dir if cfg.with_prior_preservation else None,
            class_prompt=cfg.datasets.class_prompt,
            class_num=cfg.num_class_images,
            tokenizer=tokenizer,
            size=cfg.models.resolution,
            center_crop=cfg.datasets.center_crop,
            encoder_hidden_states=pre_computed_encoder_hidden_states,
            class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
            tokenizer_max_length=cfg.tokenizer_max_length,
            image_interpolation_mode = cfg.models.image_interpolation_mode,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: dreambooth_collate_fn(examples, cfg.with_prior_preservation),
            num_workers=cfg.dataloader_num_workers,
        )

        nft_dataset = DreamBoothDataset(
            instance_data_root=unsafe_instance_dir,
            instance_prompt=unsafe_instance_prompt,
            # class_data_root=class_data_dir if cfg.with_prior_preservation else None,
            class_data_root=None,
            # class_prompt=cfg.datasets.class_prompt,
            # class_num=cfg.num_class_images,
            tokenizer=tokenizer,
            size=cfg.models.resolution,
            center_crop=cfg.datasets.center_crop,
            encoder_hidden_states=pre_computed_encoder_hidden_states,
            class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
            tokenizer_max_length=cfg.tokenizer_max_length,
            image_interpolation_mode = cfg.models.image_interpolation_mode,
        )
        nft_dataloader = torch.utils.data.DataLoader(
            nft_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: dreambooth_collate_fn(examples, False),
            num_workers=cfg.dataloader_num_workers,
        )

        # nft_dataset = None
        # nft_dataloader = None
    return train_dataset, train_dataloader, nft_dataset, nft_dataloader

class PromptDataset(torch.utils.data.Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

class DreamBoothDataset(torch.utils.data.Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        image_interpolation_mode=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = pathlib.Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(pathlib.Path(instance_data_root).iterdir())

        removable = []
        for i in range(0,len(self.instance_images_path)):
            if str(self.instance_images_path[i]).endswith(".json"):
                removable.append(self.instance_images_path[i])
        for val in removable:
            self.instance_images_path.remove(val)

        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = pathlib.Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())



            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        interpolation = getattr(transforms.InterpolationMode, image_interpolation_mode.upper(), None)
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation mode {interpolation=}.")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=interpolation),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


def dreambooth_collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch

if __name__ == "__main__":
    # 1. 测试prepare_dataset能否加载两个nonfinetunable dataset和正常的train dataset。
    #
    #    测试任务：使用pdb进入调试模型保存一下两个数据集的图片和prompt看看。
    # stable diffusion位置： /data1/dif/sd1-5
    # datasets位置: /data1/dif/non-finetune-lora/test_case/target
    import dataclasses
    from transformers import AutoTokenizer
    @dataclasses.dataclass
    class DATASETS:
        class_prompt: str="Hello xx world"
        center_crop: bool = True
        instance_prompt: str="Hello world"
    
    @dataclasses.dataclass
    class MODELS:
        image_interpolation_mode="lanczos"
        resolution=512

    @dataclasses.dataclass
    class CFG:
        datasets: DATASETS
        models: MODELS
        tokenizer_max_length=152
        num_class_images=100
        with_prior_preservation=True
        train_batch_size=5
        dataloader_num_workers=0
    tokenizer = AutoTokenizer.from_pretrained(
        "/data1/dif/sd1-5", subfolder="tokenizer", use_fast=False
    )
    cfg = CFG(datasets=DATASETS(), models=MODELS())
    train_dataset, train_dataloader, nft_dataset, nft_dataloader = prepare_dataset(
        cfg,
        dataset_name='/data1/dif/non-finetune-lora/test_case/target/0',
        cache_dir=None,
        tokenizer=tokenizer,
        accelerator=None,
        use_dream_booth=True,
        class_data_dir=None,
        pre_computed_encoder_hidden_states = None,
        pre_computed_class_prompt_encoder_hidden_states = None,
    )
    import pdb;pdb.set_trace()