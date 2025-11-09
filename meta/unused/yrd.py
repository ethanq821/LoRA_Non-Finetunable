#!/usr/bin/env python3

'''
We have nothing to do with learn2learn.
'''
import re
import torch
from torch import nn
from torch import optim
from torch.autograd import grad
from torch import Tensor
from typing import cast, Dict, Tuple, List, Union, Optional, Iterable, Any
from torch.optim.adamw import AdamW, adamw
from torch.optim.sgd import SGD, sgd


from peft.tuners.tuners_utils import BaseTunerLayer
from peft import LoraConfig

import gc
import torch.nn.functional as F
from diffusers.training_utils import compute_snr
from torch.autograd import grad


def replace_lora_layers(model: nn.Module) -> dict:
    """
    Register the lora layers with batched weights and enabling gradient propagation.
    New attributes will be added to the LoraLayers, including lora_A_tensor and lora_B_tensor

    Args:
        model (nn.Module): the base model from PeftModel to be modified.

        attention_types(List[str]): the attention types to be replaced.
        
    Returns:
        dict: the dict contained pointer used to apply modification.
        
    """

    model_lora_layers = {}
    active_names = model.active_adapters()
    assert len(active_names) == 1, "Only one adapter is allowed in MAML"
    for n, m in model.named_modules():  # 每一层都换吗？
        if isinstance(m, BaseTunerLayer):
            if hasattr(m, "set_adapter"):
                pattern = r"\.(\d+)"
                replacement = r"[\1]"
                layer_to_replace = f'model.{re.sub(pattern, replacement, n)}'
                layer_to_replace_A = f'{layer_to_replace}.lora_A[{active_names[0]}].weight'
                layer_to_replace_B = f'{layer_to_replace}.lora_B[{active_names[0]}].weight'
                model_lora_layers[layer_to_replace_A] = eval(layer_to_replace_A)
                model_lora_layers[layer_to_replace_B] = eval(layer_to_replace_B)

    return model_lora_layers

def set_loras(
    layer_dict: dict,
    model: nn.Module,
    attention_types: List[str],
):
    """
    Apply lora to the model for saving.

    Args:
        layer_dict (dict): 
            The dict containing pointer to apply lora or models. Be like {model.layers.0.self_attn.q_proj: LoraLayer..}
            Used in training for gradient propagation scenarios, should call 'replace_lora_layers' function first before execution.
            If the layer_dict is None, the 'model' arg must be specified.

        model (nn.Module): 
            The peft model to replace. 
            Replace the lora in the target model will stop the gradient propagation, but can be save and load through standard peft behaviour. 
            If the model is None, the 'layer_dict' arg must be specified.

        attention_types (List[str]):
            The attention types to apply lora. Be like ['q_proj', 'k_proj', 'v_proj', 'out_proj'].
            
    """
    # modification 1.
    # for name in model.base_model.targeted_module_names:
    for name in model.targeted_module_names:
        
        # modification 2.
        # if "q_proj" in name or "v_proj" in name:
        if name.split(".")[-2] in attention_types:

            # modification 3.
            # layer_to_replace = get_nested_module(model.base_model.model, name)
            layer_to_replace = eval(model+"."+name)

            # modification 4.
            # layer_to_replace.lora_A_tensor = layer_to_replace.lora_A[model.active_adapter].weight.T.unsqueeze(0).clone()
            # layer_to_replace.lora_B_tensor = layer_to_replace.lora_B[model.active_adapter].weight.T.unsqueeze(0).clone()
            layer_to_replace.lora_A[model.active_adapter].weight = layer_to_replace.lora_A_tensor.squeeze(0).clone().T
            layer_to_replace.lora_B[model.active_adapter].weight = layer_to_replace.lora_B_tensor.squeeze(0).clone().T

def clone(params, clear_grad=False, clone_grad=False):
    if not type(params) == list:
        params = list(params)
    new_params = []
    for param in params:
        if not clone_grad:
            cloned = {k: v.clone() for k, v in param.items()}
            if clear_grad:
                for k, v in cloned.items():
                    v.grad = 0.0
        else:
            cloned = {k: v.grad.data for k, v in param.items()}
        new_params.append(cloned)
    return new_params

class MetaAdamW(AdamW):
    def step(self):
        self._cuda_graph_capture_health_check()

        new_params = []
        for p, grad in zip(self.param_groups[0]["params"], grads):
            pn = torch.zeros_like(p.data)#.requires_grad_()
            pn.data = p.data
            pn.grad = grad
            new_params.append(pn)
        self.param_groups[0]["params"] = new_params

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            amsgrad: bool = group["amsgrad"]
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
            )

        return self.param_groups[0]["params"]

class MetaSGD(SGD):
    def step(self, grads):

        new_params = []
        for p, grad in zip(self.param_groups[0]["params"], grads):
            # pn = torch.zeros_like(p.data)#.requires_grad_()
            # pn.data = p.data
            pn = p.clone()
            pn.grad = grad
            new_params.append(pn)
        self.param_groups[0]["params"] = new_params
        
        # import pdb;pdb.set_trace()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return self.param_groups[0]["params"]



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

def training_loss(
    unet_in_channels,
    batch,
    vae,
    noise_scheduler,
    text_encoder,
    unet,
    cfg,
    weight_dtype,
    device,
    nft_dataloader=None,
    all_lora_layers=None,
    second_order=None,
    allow_unused=None
):
    if not cfg.losses.is_normal:
        if cfg.losses.alg.lower() == 'maml':
            normal_loss, all_nft_loss = maml(unet_in_channels, batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype, device, nft_dataloader, all_lora_layers, second_order, allow_unused)
        elif cfg.losses.alg.lower() == 'reptile':
            normal_loss, all_nft_loss = reptile(unet_in_channels, batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype, device, nft_dataloader, all_lora_layers, second_order, allow_unused)
    else:
        normal_loss = mse_loss(unet_in_channels, batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype)
        all_nft_loss = torch.tensor(0.0).to(device, dtype=weight_dtype)
    return normal_loss, all_nft_loss

def mse_loss(
    unet_in_channels,
    batch,
    vae,
    noise_scheduler,
    text_encoder,
    unet,
    cfg,
    weight_dtype,
    latents=None,
):
    # Get the text embedding for conditioning
    # 文本 -> embedding：text_encoder 处理输入文本，供条件模型使用。
    # encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
    # if cfg.pre_compute_text_embeddings:
    #######这里被大改了#########
    encoder_hidden_states = batch["input_ids"]
    # else:
        # encoder_hidden_states = encode_prompt(
        #     text_encoder,
        #     batch["input_ids"],
        #     batch["attention_mask"],
        #     text_encoder_use_attention_mask=cfg.text_encoder_use_attention_mask,
        # )

    # Convert images to latent space
    if latents is None:
        latents = batch["pixel_values"]
    # 图像 -> latent：vae 对像素图片编码至潜空间 latent。
    latents = vae.encode(latents.to(dtype=weight_dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    # 噪声生成：随机噪声 + 可选噪声偏移。
    noise = torch.randn_like(latents)
    if cfg.noise_offset:
        noise += cfg.noise_offset * torch.randn(
            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
        )

    # bsz = latents.shape[0]
    bsz, channels, height, width = latents.shape
    # torch.randint : 均匀分布
    # torch.randn : 正态分布
    # 采样时间步 t：随机选择扩散步骤。
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
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            loss = loss + cfg.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

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
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="none")
            loss = loss + cfg.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
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
    allow_unused
):
    # currently all_lora_layers : [{k:v},{k:v}]
    if not second_order:
        second_order = cfg.losses.second_order
    if not allow_unused:
        allow_unused = cfg.losses.allow_unused

    # test 1. 递推合并顺序？
    lora_values = [v for d in all_lora_layers for k, v in d.items()]
    backup_lora_values = [v for d in all_lora_layers for k, v in d.items()]
    backup_grad_values = [v.grad.clone().detach() for d in all_lora_layers for k, v in d.items()]
    for v in lora_values:
        v.grad = 0.0

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
                latents=torch.zeros_like(batch["pixel_values"]) #############直接让预测的噪声归零还是让预测的图片为黑色？
            )
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

    return mse_loss(unet_in_channels, batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype), all_nft_loss

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
):
    # precalculate the normal loss
    normal_loss = mse_loss(unet_in_channels, batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype)
    
    lora_values = []
    backup_lora_values = []
    backup_grad_values = []
    for d in all_lora_layers:
        for k, v in d.items():
            lora_values.append(v)
            backup_lora_values.append(v.clone().detach())
            backup_grad_values.append(v.grad.clone().detach())
            v.grad = 0.0

    all_nft_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if cfg.losses.optim == "adam":
        nft_optimizer = AdamW(lora_values, lr=cfg.losses.nft_learning_rate, weight_decay=0.0)
    elif cfg.losses.optim == "sgd":
        nft_optimizer = SGD(lora_values, lr=cfg.losses.nft_learning_rate, weight_decay=0.0)
    else:
        raise NotImplementedError
    for epoch in range(0, cfg.losses.num_nft_epochs):
        for step, nft_batch in enumerate(nft_dataloader):
            nft_loss = mse_loss(unet_in_channels, nft_batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype,
                latents=torch.zeros_like(batch["pixel_values"]) #############直接让预测的噪声归零还是让预测的图片为黑色？
            )
            nft_loss.backward()
            nft_optimizer.step()
            all_nft_loss += nft_loss

    # reptile update
    for v, v_bak, v_bak_grad in zip(lora_values, backup_lora_values, backup_grad_values):
        v.data = v_bak - cfg.losses.beta * (v.data - v_bak.data)
        v.grad.data = backup_grad[k].data

    del backup_lora_values
    del backup_grad_values
    gc.collect()

    return normal_loss, all_nft_loss

def reptile_for_test(
    unet_in_channels,
    batch,
    nft_batch,
    vae,
    noise_scheduler,
    text_encoder,
    unet,
    
    weight_dtype,
    device,
    # nft_dataloader,
    all_lora_layers,
):
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
    # for epoch in range(0, cfg.losses.num_nft_epochs):
    #     for step, nft_batch in enumerate(nft_dataloader):
    nft_loss = mse_loss(unet_in_channels, nft_batch, vae, noise_scheduler, text_encoder, unet, cfg, weight_dtype,
        latents=torch.zeros_like(batch["pixel_values"]) #############直接让预测的噪声归零还是让预测的图片为黑色？
    )
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

def create_or_replace_lora(
    is_normal: bool,
    model: torch.nn.Module,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    init_lora_weights: str,
    target_modules: List[str],
):
    
    try:
        adapter_names = model.active_adapters()
    except Exception:
        adapter_names = None

    if adapter_names:
        assert len(adapter_names) == 1, "Cannot create LoRA when active adapters are more than 1"
        delete_adapters(model, adapter_names)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=r,
        lora_dropout=lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    model.add_adapter(lora_config)
    lora_layers = None
    if not is_normal:
        lora_layers = replace_lora_layers(model)
    return lora_layers


# copied from diffusers/transformers lib
def delete_adapters(model, adapter_names):
    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]
    for adapter_name in adapter_names:
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "delete_adapter"):
                    module.delete_adapter(adapter_name)
                else:
                    raise ValueError(
                        "The version of PEFT you are using is not compatible, please use a version that is greater than 0.6.1"
                    )

        # For transformers integration - we need to pop the adapter from the config
        if getattr(model, "_hf_peft_config_loaded", False) and hasattr(model, "peft_config"):
            model.peft_config.pop(adapter_name, None)
            # In case all adapters are deleted, we need to delete the config
            # and make sure to set the flag to False
            if len(model.peft_config) == 0:
                del model.peft_config
                model._hf_peft_config_loaded = None
                
        if hasattr(model, "peft_config"):
            model.peft_config.pop(adapter_name, None)

if __name__ == "__main__":
    # 1. 测试nft_loss的reptile损失能否正确更新内循环，并且更新完成之后只影响lora权重的梯度而非lora权重的大小，同时由于使用了指针，模型内部的权重也需要得到正确更新，这样就不会对下一个循环中产生影响。
    # 比较麻烦，需要使用类似replace_lora_layers的思路抓取模型的权重，先放着

    # test the meta optimizer
    # meta_params = (torch.tensor([0.1], requires_grad=True), torch.tensor([0.4], requires_grad=True))
    # # meta_optimizer = MetaSGD(meta_params, lr=0.01, differentiable=True)
    # params = [p.clone() for p in meta_params]
    # grad_container = []
    # # import pdb;pdb.set_trace()
    # # operate 
    # # for i in range(1):
    # #     meta_params = [param for param in params]
    # #     loss = (meta_params[0] * meta_params[1]).sum()
    # #     grad = torch.autograd.grad(loss, meta_params, create_graph=True,retain_graph=True)
    # #     # import pdb;pdb.set_trace()
    # #     meta_params = meta_optimizer.step(grad)
    # #     grad = None
    # # print(meta_params)
    # # final_loss = (meta_params[0] + meta_params[1]).sum()
    # # grad = torch.autograd.grad(final_loss, params)
    # # print(grad)
    # def easy_model(meta_params):
    #     loss = (meta_params[0] * meta_params[1]).sum()
    #     grad = torch.autograd.grad(loss, meta_params, create_graph=True,retain_graph=True)
    #     # second_grad = torch.autograd.grad(sum(grad), meta_params)
    #     return grad
    # import pdb;pdb.set_trace()
    # jcb = torch.autograd.functional.jacobian(easy_model, meta_params)
    # import pdb;pdb.set_trace()
    # # grad_container.append(1. - 0.01 * second_grad)
    # for g in second_grad:
    #     g_add = 1. - 0.01 * g
    #     grad_container.append(g_add)

    # # new_p = []
    # # for p,g in zip(meta_params,grad):
    # #     new_p.append((p - g * 0.01).requires_grad_())
    # # for p,g in zip(meta_params,grad):
    # meta_params=[(p - g * 0.01).detach().requires_grad_() for p, g in zip(meta_params, grad)]
    # # final_loss = (new_p[0] * new_p[1]).sum()
    # final_loss = (meta_params[0] * meta_params[1]).sum()
    # grad = torch.autograd.grad(final_loss, meta_params)
    # import pdb;pdb.set_trace()
    # # (tensor([0.3980]), tensor([0.0920]))
    
    

    
    # batch,
    # vae,
    # noise_scheduler,
    # text_encoder,
    # unet,
    # cfg,
    # weight_dtype,
    # device,
    # nft_dataloader,
    # all_lora_layers,
    # second_order,
    # allow_unused
    from hydra import initialize_config_dir, compose

    with initialize_config_dir(config_dir="/home/dev/zbq/lora_nft_Oct/lora_nonfinetunable/cfg", job_name="test", version_base=None):
        cfg = compose(config_name="config", overrides=["losses=nft"])
    # import pdb;pdb.set_trace()
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained("/data1/dif/sd1-5", torch_dtype=torch.bfloat16).to("cpu")
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    noise_scheduler = pipe.scheduler
    unet_in_channels = unet.config.in_channels
    fake_batch = {
        "input_ids": torch.randn(1, 77, 768).to("cpu", dtype=torch.bfloat16),
        "pixel_values": torch.randn(1, 3, 512, 512).to("cpu", dtype=torch.bfloat16),
    }
    fake_nft_batch = {
        "input_ids": torch.randn(1, 77, 768).to("cpu", dtype=torch.bfloat16),
        "pixel_values": torch.randn(1, 3, 512, 512).to("cpu", dtype=torch.bfloat16),
    }

    all_lora_layers = []
    unet_lora_layers = create_or_replace_lora(False, unet, 8, 8, 0, 'gaussian', ["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"])
    all_lora_layers.append(unet_lora_layers)
    # if train_text_encoder:
    text_encoder_lora_layers = create_or_replace_lora(False, text_encoder, 8, 8, 0, 'gaussian', ["q_proj", "k_proj", "v_proj", "out_proj"])
    all_lora_layers.append(text_encoder_lora_layers)
    from torch.optim import AdamW
    lora_values = []
    for d in all_lora_layers:
        for k, v in d.items():
            lora_values.append(v)
    optimizer = AdamW(lora_values, lr=1e-3)
    all_nft_loss = reptile_for_test(
        unet_in_channels,
        batch=fake_batch,
        nft_batch=fake_nft_batch,
        vae=vae,
        noise_scheduler=noise_scheduler,
        text_encoder=text_encoder,
        unet=unet,
        weight_dtype=torch.bfloat16,
        device="cpu",
        # nft_dataloader,
        all_lora_layers=all_lora_layers,
    )
    # normal_loss.backward()
    optimizer.step()
    print(f"The all nft loss is {all_nft_loss}")
    all_nft_loss = reptile_for_test(
        unet_in_channels,
        batch=fake_batch,
        nft_batch=fake_nft_batch,
        vae=vae,
        noise_scheduler=noise_scheduler,
        text_encoder=text_encoder,
        unet=unet,
        weight_dtype=torch.bfloat16,
        device="cpu",
        # nft_dataloader,
        all_lora_layers=all_lora_layers,
    )
    
    print(f"The all nft loss is {all_nft_loss}")
    # import pdb;pdb.set_trace()