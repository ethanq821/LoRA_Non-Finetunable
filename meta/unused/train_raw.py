import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorboard
import math
import pathlib
import shutil

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import logging
from contextlib import nullcontext
from tqdm.auto import tqdm

import datasets
import transformers
import diffusers


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import hydra
from omegaconf import DictConfig, OmegaConf


from peft.utils import get_peft_model_state_dict

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import convert_state_dict_to_diffusers, is_wandb_available
from diffusers.training_utils import compute_snr

from data import prepare_image, prepare_dataset
from model import prepare_model
from losses import training_loss

to_pil = T.ToPILImage()
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
logger = get_logger(__name__, log_level="INFO")

if is_wandb_available():
    import wandb

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

# def collate_fn(examples):
#     pixel_values = torch.stack([example["pixel_values"] for example in examples])
#     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
#     input_ids = torch.stack([example["input_ids"] for example in examples])
#     return {"pixel_values": pixel_values, "input_ids": input_ids}

def log_validation(
    pipeline,
    cfg,
    accelerator,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {cfg.num_validation_images} images with prompt:"
        f" {cfg.datasets.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if cfg.seed is not None:
        generator = generator.manual_seed(cfg.seed)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(cfg.num_validation_images):
            images.append(pipeline(cfg.datasets.validation_prompt, num_inference_steps=30, generator=generator).images[0])

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {cfg.datasets.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
    return images


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def train(cfg: DictConfig)-> None:

    logging_dir = pathlib.Path(cfg.output_dir, cfg.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(OmegaConf.to_yaml(cfg), main_process_only=True)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    noise_scheduler, tokenizer, text_encoder, vae, unet, weight_dtype = prepare_model(
        pretrained_model_name_or_path=cfg.models.pretrained_model_name_or_path,
        revision=cfg.models.revision,
        variant=cfg.models.variant,
        accelerator=accelerator,
        mixed_precision=cfg.mixed_precision,
        gradient_checkpointing=cfg.gradient_checkpointing,
        rank=cfg.rank,
        lora_alpha=cfg.lora_alpha,
        init_lora_weights=cfg.init_lora_weights,
        target_modules=cfg.target_modules,
    )
    # Initialize the optimizer
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer = optimizer_cls(
        lora_layers,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )
    if not cfg.losses.is_normal:
        nft_optimizer = optimizer_cls(
            lora_layers,
            lr=cfg.losses.nft_learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            weight_decay=cfg.adam_weight_decay,
            eps=cfg.adam_epsilon,
        )

    train_dataset, train_dataloader = prepare_dataset(
        cfg,
        tokenizer,
        accelerator,
    )
    """
    BUGGGGGGGGGG!!!!!!!!! It will be normal dataset
    """
    if not cfg.losses.is_normal:
        nft_dataset, nft_dataloader = prepare_dataset(
            cfg,
            tokenizer,
            accelerator,
        )

    num_warmup_steps_for_scheduler = cfg.lr_warmup_steps * accelerator.num_processes
    if cfg.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / cfg.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            cfg.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = cfg.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )
    if not cfg.losses.is_normal:
        # num_warmup_steps_for_scheduler = cfg.lr_warmup_steps * accelerator.num_processes
        num_warmup_steps_for_nft_scheduler = 0
        if cfg.max_train_steps is None:
            len_nft_dataloader_after_sharding = math.ceil(len(nft_dataloader) / accelerator.num_processes)
            num_update_steps_per_epoch = math.ceil(len_nft_dataloader_after_sharding / cfg.gradient_accumulation_steps)
            num_training_steps_for_nft_scheduler = (
                cfg.losses.num_nft_epochs * num_update_steps_per_epoch * accelerator.num_processes
            )
        else:
            num_training_steps_for_nft_scheduler = cfg.max_train_steps * accelerator.num_processes

        nft_scheduler = get_scheduler(
            cfg.losses.nft_scheduler,
            optimizer=nft_optimizer,
            num_warmup_steps=num_warmup_steps_for_nft_scheduler,
            num_training_steps=num_training_steps_for_nft_scheduler,
        )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    if not cfg.losses.is_normal:
        unet, optimizer, nft_optimizer, train_dataloader, nft_dataloader, lr_scheduler, nft_scheduler = accelerator.prepare(
            unet, optimizer, nft_optimizer, train_dataloader, nft_dataloader, lr_scheduler, nft_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != cfg.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    """
    BUGGGGGGGGGG!!!!!!!!!
    """
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-nft-training", config={k:v for k, v in OmegaConf.to_container(cfg).items() if type(v) not in [list, dict]})

    # Train!
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    for epoch in range(first_epoch, cfg.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                loss = training_loss(
                    batch,
                    vae,
                    noise_scheduler,
                    text_encoder,
                    unet,
                    cfg,
                    weight_dtype,
                    nft_optimizer if not cfg.losses.is_normal else None,
                    nft_dataloader if not cfg.losses.is_normal else None,
                    nft_scheduler if not cfg.losses.is_normal else None,
                    # lora_layers,
                )
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if cfg.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(cfg.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= cfg.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - cfg.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(cfg.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = unwrap_model(unet, accelerator)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

        if accelerator.is_main_process:
            if cfg.datasets.validation_prompt is not None and epoch % cfg.validation_epochs == 0:
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    cfg.models.pretrained_model_name_or_path,
                    unet=unwrap_model(unet, accelerator),
                    revision=cfg.models.revision,
                    variant=cfg.models.variant,
                    torch_dtype=weight_dtype,
                )
                images = log_validation(pipeline, cfg, accelerator, epoch)

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet, accelerator)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=cfg.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        # Final inference
        # Load previous pipeline
        if cfg.datasets.validation_prompt is not None:
            pipeline = DiffusionPipeline.from_pretrained(
                cfg.models.pretrained_model_name_or_path,
                revision=cfg.models.revision,
                variant=cfg.models.variant,
                torch_dtype=weight_dtype,
            )

            # load attention processors
            pipeline.load_lora_weights(cfg.output_dir)

            # run inference
            images = log_validation(pipeline, cfg, accelerator, epoch, is_final_validation=True)

    accelerator.end_training()

if __name__ == "__main__":
    train()