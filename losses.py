import gc
import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_snr

from data import encode_prompt
from model import unwrap_model

from meta.lora_maml import MetaAdamW, MetaSGD#, clone
from torch.optim import AdamW, SGD
from torch.autograd import grad
from torchvision.utils import save_image

# def training_loss(
#     unet_in_channels,
#     batch,
#     vae,
#     noise_scheduler,
#     text_encoder,
#     unet,
#     cfg,
#     weight_dtype,
#     device,
#     nft_dataloader=None,
#     all_lora_layers=None,
#     second_order=None,
#     allow_unused=None
# ):
#     # if not cfg.losses.is_normal:
#         # if cfg.losses.alg.lower() == 'maml':
#             # normal_loss, all_nft_loss = maml(unet_in_channels, batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype, device, nft_dataloader, all_lora_layers, second_order, allow_unused)
#         # elif cfg.losses.alg.lower() == 'reptile':
#             # normal_loss, all_nft_loss = reptile(unet_in_channels, batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype, device, nft_dataloader, all_lora_layers, second_order, allow_unused)
#     # else:
#     normal_loss = mse_loss(unet_in_channels, batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype)
#         # all_nft_loss = torch.tensor(0.0).to(device, dtype=weight_dtype)
#     return normal_loss#, all_nft_loss

def mse_loss(
    unet_in_channels,
    batch,
    vae,
    noise_scheduler,
    text_encoder,
    unet,
    cfg,
    weight_dtype,
    target_mode=None,
    noise=None,     #use only in debugging
    timesteps=None, #use only in debugging
):
    # Get the text embedding for conditioning
    # 文本 -> embedding：text_encoder 处理输入文本，供条件模型使用。
    # encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
    # print(f'The batch image shape is {batch['pixel_values'].shape}, The batch text shape is {batch["input_ids"].shape}')
    if cfg.pre_compute_text_embeddings:
        encoder_hidden_states = batch["input_ids"]
    else:
        encoder_hidden_states = encode_prompt(
            text_encoder,
            batch["input_ids"],
            batch["attention_mask"],
            text_encoder_use_attention_mask=cfg.text_encoder_use_attention_mask,
        )

    # Convert images to latent space
    # if latents is None:
    latents = batch["pixel_values"]
        
    # 图像 -> latent：vae 对像素图片编码至潜空间 latent。
    latents = vae.encode(latents.to(dtype=weight_dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    # 噪声生成：随机噪声 + 可选噪声偏移。
    # if noise is None:
    noise = torch.randn_like(latents)
    if cfg.noise_offset:
        noise += cfg.noise_offset * torch.randn(
            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
        )
    # print(f'the noise is {noise}')
    # bsz = latents.shape[0]
    bsz, channels, height, width = latents.shape
    # torch.randint : 均匀分布
    # torch.randn : 正态分布
    # 采样时间步 t：随机选择扩散步骤。
    # if timesteps is None:
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()
    # 加噪：使用噪声调度器将 latent 加噪到 noisy_latents。
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # 目标设定：根据预测类型选择要拟合的 target（噪声或 velocity）。
    if cfg.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=cfg.prediction_type)

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    # if unwrap_model(unet, accelerator).config.in_channels == channels * 2:
    if unet_in_channels == channels * 2:
        noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

    # Predict the noise residual and compute loss
    # 模型预测：unet 输入 noisy_latents、timesteps、文本条件，输出 model_pred。
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

    # if model predicts variance, throw away the prediction. we will only train on the
    # simplified training objective. This means that all schedulers using the fine tuned
    # model must be configured to use one of the fixed variance variance types.
    if model_pred.shape[1] == 6:
        model_pred, _ = torch.chunk(model_pred, 2, dim=1)


    # 损失计算：默认的 MSE，或基于 SNR 的加权 MSE。
    if cfg.snr_gamma is None:
        if cfg.with_prior_preservation:
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            # if target_mode=="reverse":
                # loss = F.mse_loss(-model_pred.float(), target.float(), reduction="mean")
            # elif target_mode == None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # else:
                # raise NotImplementedError(f"Unknown target mode: {target_mode}")
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            loss = loss + cfg.prior_loss_weight * prior_loss
        else:
            # if target_mode=="reverse":
                # loss = F.mse_loss(-model_pred.float(), target.float(), reduction="mean")
            # elif target_mode == None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # else:
            #     raise NotImplementedError(f"Unknown target mode: {target_mode}")

    else:
        # raise NotImplementedError("SNR loss is not implemented yet.")
        # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
            dim=1
        )[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)
            
        # 返回 loss：供优化器反向传播，更新模型参数。
        if cfg.with_prior_preservation:
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            # loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            
            # if target_mode=="reverse":
            #     loss = F.mse_loss(-model_pred.float(), target.float(), reduction="mean")
            # elif target_mode == None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # else:
            #     raise NotImplementedError(f"Unknown target mode: {target_mode}")
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="none")
            loss = loss + cfg.prior_loss_weight * prior_loss
        else:
            # loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            
            # if target_mode=="reverse":
            #     loss = F.mse_loss(-model_pred.float(), target.float(), reduction="mean")
            # elif target_mode == None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # else:
            #     raise NotImplementedError(f"Unknown target mode: {target_mode}")
        # loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()
    return loss

def maml(
    unet_in_channels,
    batch,
    vae,
    noise_scheduler,
    text_encoder,
    unet,
    cfg,
    weight_dtype,
    device,
    nft_dataloader,
    all_lora_layers,
    second_order,
    allow_unused,
    noise=None,
    timesteps=None,
):
    # currently all_lora_layers : [{k:v},{k:v}]
    
    if not second_order:
        second_order = cfg.losses.second_order
    if not allow_unused:
        allow_unused = cfg.losses.allow_unused

    lora_values = []
    backup_lora_values = []
    backup_grad_values = []
    for d in all_lora_layers:
        for k, v in d.items():
            lora_values.append(v)
            backup_lora_values.append(v.clone().detach())
            backup_grad_values.append(v.grad.clone().detach()) if v.grad is not None else backup_grad_values.append(torch.zeros_like(v))
            v.grad = torch.zeros_like(v.grad) if v.grad is not None else None

    all_nft_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if cfg.losses.optim == "adam":
        nft_optimizer = MetaAdamW(lora_values, lr=cfg.losses.nft_learning_rate, weight_decay=0.0)
    elif cfg.losses.optim == "sgd":
        nft_optimizer = MetaSGD(lora_values, lr=cfg.losses.nft_learning_rate, weight_decay=0.0)
    else:
        raise NotImplementedError
    for epoch in range(0, cfg.losses.num_nft_epochs):
        for step, nft_batch in enumerate(nft_dataloader):
            nft_loss = mse_loss(unet_in_channels, nft_batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype,
                target_mode=None,noise=noise,timesteps=timesteps
            )#############让预测的图片为仍然为正常图片，但是下面更新lora权重的时候反方向更新
            # Backpropagate
            grad = grad(nft_loss, lora_values,retain_graph=cfg.second_order,create_graph=cfg.second_order,allow_unused=cfg.allow_unused)
            # gradient clipping is not implemented!!!
            lora_values = nft_optimizer.step(grad) # create a new dict of lora layers
            all_nft_loss += nft_loss

    grad = grad(nft_loss, backup_lora_values)
    for v, v_bak, v_bak_grad in zip(lora_values, backup_lora_values, backup_grad_values):
        v.data = v_bak.data
        v.grad.data = cfg.losses.beta * v_bak.grad.data + backup_grad[k].data

    del backup_lora_values
    del backup_grad_values
    gc.collect()

    return all_nft_loss

def reptile(
    unet_in_channels,
    batch,
    vae,
    noise_scheduler,
    text_encoder,
    unet,
    cfg,
    weight_dtype,
    device,
    nft_dataloader,
    all_lora_layers,
    second_order=None, # useless
    allow_unused=None, # useless
    noise=None,
    timesteps=None,
):
    # print("Reptile Mode")
    lora_values = []
    backup_lora_values = []
    backup_grad_values = []
    for d in all_lora_layers:
        for k, v in d.items():
            lora_values.append(v)
            backup_lora_values.append(v.clone().detach())
            backup_grad_values.append(v.grad.clone().detach()) if v.grad is not None else backup_grad_values.append(torch.zeros_like(v))
            v.grad = torch.zeros_like(v.grad) if v.grad is not None else None

    all_nft_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if cfg.losses.optim == "adamw":
        nft_optimizer = AdamW(lora_values, lr=cfg.losses.nft_learning_rate, weight_decay=0.0)
    elif cfg.losses.optim == "sgd":
        nft_optimizer = SGD(lora_values, lr=cfg.losses.nft_learning_rate, weight_decay=0.0)
    else:
        raise NotImplementedError
    for epoch in range(0, cfg.losses.num_nft_epochs):
        for step, nft_batch in enumerate(nft_dataloader):
            nft_loss = mse_loss(unet_in_channels, nft_batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype,
                target_mode=None, noise=noise, timesteps=timesteps 
            )#############让预测的图片为仍然为正常图片，但是下面更新lora权重的时候反方向更新
            nft_loss.backward()
            nft_optimizer.step()
            all_nft_loss += nft_loss

    # reptile update
    for v, v_bak, v_bak_grad in zip(lora_values, backup_lora_values, backup_grad_values):
        v.data = v_bak - cfg.losses.beta * (v.data - v_bak.data)
        v.grad = v_bak_grad if v.grad is not None else None

    del backup_lora_values
    del backup_grad_values
    gc.collect()

    return all_nft_loss

if __name__ == "__main__":
    # 1. 测试nft_loss的reptile损失能否正确更新内循环，并且更新完成之后只影响lora权重的梯度而非lora权重的大小，同时由于使用了指针，模型内部的权重也需要得到正确更新，这样就不会对下一个循环中产生影响。
    # 比较麻烦，需要使用类似replace_lora_layers的思路抓取模型的权重，先放着
    ...