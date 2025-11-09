"""
Tuning batch datasets w/ or w/o nft, based on dreambooth.
The original scripts are provided by diffusers' dreambooth implementation.
"""

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
from torchvision.utils import save_image
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
from huggingface_hub.utils import insecure_hashlib

from peft.utils import get_peft_model_state_dict

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import convert_state_dict_to_diffusers, is_wandb_available
from diffusers.training_utils import compute_snr, free_memory, cast_training_params

from data import prepare_dataset, PromptDataset, tokenize_prompt, encode_prompt
from model import prepare_model_for_training, unwrap_model, load_base_model
from losses import mse_loss

import gc
import time
# from utils import 

to_pil = T.ToPILImage()
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
logger = get_logger(__name__, log_level="INFO")

if is_wandb_available():
    import wandb

def log_validation(
    pipeline,
    cfg,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):  
    if cfg.datasets.instance_prompt:
        validation_prompt = cfg.datasets.instance_prompt
    else:
        validation_prompt = cfg.datasets.validation_prompt
        logger.info(
            f"Running validation... \n Generating {cfg.num_validation_images} images with prompt:"
            f" {validation_prompt}."
        )

    # from dreambooth training script
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
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
            image = pipeline(**pipeline_args, generator=generator, num_inference_steps=30).images[0]
            images.append(image)

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

    # from dreambooth training script
    del pipeline
    free_memory()

    return images


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def train(cfg: DictConfig)-> None:
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_base_model(
        pretrained_model_name_or_path=cfg.models.pretrained_model_name_or_path,
        revision=cfg.models.revision,
        variant=cfg.models.variant,
    )
    fixed_noise_for_test = None
    fixed_timestep_for_test = None
    # if not cfg.eval:
    #     fixed_noise_for_test = torch.randn(1, 4, 64, 64)
    #     fixed_timestep_for_test = torch.tensor([366])
    # setting output_dir
    experiments = [name for name in os.listdir(cfg.datasets.dataset_name) if os.path.isdir(os.path.join(cfg.datasets.dataset_name, name))]
    print(f"Total Training experiments: {experiments}")
    for experiment in experiments:
        # print(f"Training experiment: {experiment}")
        
        output_dir = os.path.join(cfg.output_dir, experiment)
        dataset_name = os.path.join(cfg.datasets.dataset_name, experiment)
        class_data_dir = os.path.join(output_dir, "class_data")
        # cache_dir = os.path.join(cfg.datasets.cache_dir, experiment)

        
        logging_dir = pathlib.Path(output_dir, cfg.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
            log_with=cfg.report_to,
            project_config=accelerator_project_config,
        )
        if not cfg.eval:
            fixed_noise_for_test = fixed_noise_for_test.to(accelerator.device, dtype=torch.float16)
            fixed_timestep_for_test = fixed_timestep_for_test.to(accelerator.device, dtype=torch.float16)# cpu is fp16

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
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            # logger.info(f"Creating model folder: {model_dir}")
            logger.info(f"Training on experiment: {experiment}")

        # Generate class images if prior preservation is enabled.
        if cfg.with_prior_preservation:
            class_images_dir = pathlib.Path(class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < cfg.num_class_images:
                torch_dtype = torch.float16 if accelerator.device.type in ("cuda", "xpu") else torch.float32
                if cfg.prior_generation_precision == "fp32":
                    torch_dtype = torch.float32
                elif cfg.prior_generation_precision == "fp16":
                    torch_dtype = torch.float16
                elif cfg.prior_generation_precision == "bf16":
                    torch_dtype = torch.bfloat16
                pipeline = DiffusionPipeline.from_pretrained(
                    cfg.models.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    # safety_checker=None,
                    revision=cfg.models.revision,
                    variant=cfg.models.variant,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = cfg.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(cfg.datasets.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=cfg.sample_batch_size)

                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)
                # sample_datas = [{"prompt": cfg.class_prompt} for _ in range(num_new_images)]

                for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                ):
                    images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        
                        # 2025.11.06 added, to block the dreambooth from generating all-black images,
                        # which occurs when the safety checker decide the image is unsafe.
                        # test the all-black scenario
                        if T.ToTensor()(image).sum() == 0.: # all black image from PIL.Image to tensor is filled with 0.
                            continue

                        hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)

                del pipeline
                free_memory()
        # print("Class images generated successfully.")
        text_encoder, vae, unet, weight_dtype, all_lora_layers = prepare_model_for_training(
            cfg.losses.is_normal,
            text_encoder, 
            vae, 
            unet,
            accelerator=accelerator,
            mixed_precision=cfg.mixed_precision,
            gradient_checkpointing=cfg.gradient_checkpointing,
            train_text_encoder = cfg.train_text_encoder,
            rank=cfg.rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            init_lora_weights=cfg.init_lora_weights,
            unet_target_modules = cfg.unet_target_modules,
            text_encoder_target_modules = cfg.text_encoder_target_modules,
            logger=logger,
        )
        # print("Model prepared")
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

        # lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
        params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
        if cfg.train_text_encoder:
            params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, text_encoder.parameters()))


        optimizer = optimizer_cls(
            params_to_optimize,
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            weight_decay=cfg.adam_weight_decay,
            eps=cfg.adam_epsilon,
        )


        if cfg.pre_compute_text_embeddings:
            def compute_text_embeddings(prompt):
                with torch.no_grad():
                    text_inputs = tokenize_prompt(tokenizer, prompt)
                    prompt_embeds = encode_prompt(
                        text_encoder,
                        text_inputs.input_ids,
                        text_inputs.attention_mask,
                        text_encoder_use_attention_mask=cfg.text_encoder_use_attention_mask,
                    )
                return prompt_embeds

            pre_computed_encoder_hidden_states = compute_text_embeddings(cfg.datasets.instance_prompt)
            validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

            if cfg.datasets.instance_prompt is not None:
                validation_prompt_encoder_hidden_states = compute_text_embeddings(cfg.datasets.instance_prompt)
                # validation_prompt_encoder_hidden_states = compute_text_embeddings(cfg.datasets.validation_prompt)
            else:
                validation_prompt_encoder_hidden_states = None

            if cfg.class_prompt is not None:
                pre_computed_class_prompt_encoder_hidden_states = compute_text_embeddings(cfg.datasets.class_prompt)
            else:
                pre_computed_class_prompt_encoder_hidden_states = None

            text_encoder = None
            tokenizer = None

            gc.collect()
            free_memory()
        else:
            pre_computed_encoder_hidden_states = None
            validation_prompt_encoder_hidden_states = None
            validation_prompt_negative_prompt_embeds = None
            pre_computed_class_prompt_encoder_hidden_states = None

        train_dataset, train_dataloader, nft_dataset, nft_dataloader = prepare_dataset(
            cfg=cfg,
            dataset_name=dataset_name,
            # cache_dir,
            tokenizer=tokenizer,
            accelerator=accelerator,
            use_dream_booth=True,
            class_data_dir = class_data_dir,
            pre_computed_encoder_hidden_states=pre_computed_encoder_hidden_states,
            pre_computed_class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
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
        unet_in_channels = unet.config.in_channels
        # Prepare everything with our `accelerator`.
        if cfg.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )
        if nft_dataloader is not None:
            nft_dataloader = accelerator.prepare(nft_dataloader)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        if cfg.max_train_steps is None:
            max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
            if num_training_steps_for_scheduler != max_train_steps * accelerator.num_processes:
                logger.warning(
                    f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                    f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                    f"This inconsistency may result in the learning rate scheduler not functioning properly."
                )
        else:
            max_train_steps = max_train_steps
        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth", config={k:v for k, v in OmegaConf.to_container(cfg).items() if type(v) not in [list, dict]})

        # Train!
        total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        global_step = 0
        first_epoch = 0
        if not cfg.losses.is_normal:
            nft_func = None
            if cfg.losses.alg.lower() == "reptile":
                from losses import reptile
                nft_func = reptile
            elif cfg.losses.alg.lower() == "maml":
                from losses import maml
                nft_func = maml

        # Potentially load in the weights and states from a previous save
        if cfg.resume_from_checkpoint:
            if cfg.resume_from_checkpoint != "latest":
                path = os.path.basename(cfg.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(output_dir)
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
                accelerator.load_state(os.path.join(output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
        for epoch in range(first_epoch, num_train_epochs):
            unet.train()
            if cfg.train_text_encoder:
                text_encoder.train()
            # train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    loss = mse_loss(
                        unet_in_channels=unet_in_channels,
                        batch=batch,
                        vae=vae,
                        noise_scheduler=noise_scheduler,
                        text_encoder=text_encoder,
                        unet=unet,
                        cfg=cfg,
                        weight_dtype=weight_dtype,
                        noise=fixed_noise_for_test,
                        timesteps=fixed_timestep_for_test
                    )

                    # Gather the losses across all processes for logging (if we use distributed training).
                    # avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                    # train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if nft_func:
                        nft_loss = nft_func(
                            unet_in_channels=unet_in_channels,
                            batch=batch,
                            vae=vae,
                            noise_scheduler=noise_scheduler,
                            text_encoder=text_encoder,
                            unet=unet,
                            cfg=cfg,
                            weight_dtype=weight_dtype,
                            device=accelerator.device, 
                            nft_dataloader=nft_dataloader, 
                            all_lora_layers=all_lora_layers, 
                            second_order=cfg.losses.second_order, 
                            allow_unused=cfg.losses.allow_unused,
                            noise=fixed_noise_for_test,
                            timesteps=fixed_timestep_for_test
                        )
                    if accelerator.sync_gradients:
                        params_to_clip = params_to_optimize
                        accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    # train_loss = 0.0

                    if global_step % cfg.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if cfg.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(output_dir)
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
                                        removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)

                            # unwrapped_unet = unwrap_model(unet, accelerator)
                            # unet_lora_state_dict = convert_state_dict_to_diffusers(
                            #     get_peft_model_state_dict(unwrapped_unet)
                            # )

                            # StableDiffusionPipeline.save_lora_weights(
                            #     save_directory=save_path,
                            #     unet_lora_layers=unet_lora_state_dict,
                            #     safe_serialization=True,
                            # )

                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "nft_loss": nft_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= max_train_steps:
                    break

            if accelerator.is_main_process:
                if cfg.datasets.validation_prompt is not None and epoch % cfg.validation_epochs == 0:
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        cfg.models.pretrained_model_name_or_path,
                        unet=unwrap_model(unet, accelerator),
                        text_encoder=None if cfg.pre_compute_text_embeddings else unwrap_model(text_encoder, accelerator),
                        safety_checker=None,
                        revision=cfg.models.revision,
                        variant=cfg.models.variant,
                        torch_dtype=weight_dtype,
                    )
                    if cfg.pre_compute_text_embeddings:
                        pipeline_args = {
                            "prompt_embeds": validation_prompt_encoder_hidden_states,
                            "negative_prompt_embeds": validation_prompt_negative_prompt_embeds,
                        }
                    else:
                        if cfg.datasets.instance_prompt:
                            pipeline_args = {"prompt": cfg.datasets.instance_prompt}
                        else:
                            pipeline_args = {"prompt": cfg.datasets.validation_prompt}
                    if cfg.eval:
                        images = log_validation(pipeline, cfg, accelerator, pipeline_args, epoch, torch_dtype=weight_dtype,)

                    del pipeline
                    torch.cuda.empty_cache()

        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_unet = unwrap_model(unet, accelerator)
            unwrapped_unet = unwrapped_unet.to(torch.float32)

            # unwrapped_unet = unwrap_model(unet, accelerator)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
            if cfg.train_text_encoder:
                unwrapped_text_encoder = unwrap_model(text_encoder, accelerator)
                text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_text_encoder))
            else:
                text_encoder_state_dict = None
            
            StableDiffusionPipeline.save_lora_weights(
                save_directory=output_dir,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_state_dict,
                safe_serialization=True,
            )

            # Final inference
            # Load previous pipeline
            if cfg.datasets.validation_prompt is not None:
                pipeline = DiffusionPipeline.from_pretrained(
                    cfg.models.pretrained_model_name_or_path,
                    revision=cfg.models.revision,
                    safety_checker=None,
                    variant=cfg.models.variant,
                    torch_dtype=weight_dtype,
                )

                # load attention processors
                pipeline.load_lora_weights(output_dir)

                # run inference
                if cfg.eval:
                    images = log_validation(pipeline, cfg, accelerator, pipeline_args, epoch, torch_dtype=weight_dtype, is_final_validation=True)

        # prepare for next iteration
        accelerator.end_training()
        unet = unwrap_model(unet, accelerator)
        text_encoder = unwrap_model(text_encoder, accelerator)
        vae = unwrap_model(vae, accelerator)
        del accelerator
        gc.collect()

if __name__ == "__main__":
    train()
    # 1. 测试训练结束后能否正确保存权重，上一次运行的时候这里卡住了。
    #
    #    测试任务：跑一遍这个脚本并且不出错。
    #    提示：可以参考/scripts/dream_booth_normal.sh和/scripts/non_finetune.sh来进行启动