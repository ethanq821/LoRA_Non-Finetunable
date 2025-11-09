#!/usr/bin/env python3

'''
We have nothing same with learn2learn.
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


def replace_lora_layers(model: nn.Module) -> dict:
    """
    Register the lora layers with batched weights and enabling gradient propagation.
    New attributes will be added to the LoraLayers, including lora_A_tensor and lora_B_tensor

    Args:
        model (nn.Module): the base model from PeftModel to be modified.
        
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
                layer_to_replace_A = f'{layer_to_replace}.lora_A["{active_names[0]}"].weight'
                layer_to_replace_B = f'{layer_to_replace}.lora_B["{active_names[0]}"].weight'
                model_lora_layers[layer_to_replace_A] = eval(layer_to_replace_A)
                model_lora_layers[layer_to_replace_B] = eval(layer_to_replace_B)

    return model_lora_layers

class MetaAdamW(AdamW):
    def step(self):
        self._cuda_graph_capture_health_check()

        new_params = []
        for p, grad in zip(self.param_groups[0]["params"], grads):
            np = torch.zeros_like(p.data)
            np.data = p.data
            np.grad = grad
            new_params.append(np)
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
            np = torch.zeros_like(p.data)
            np.data = p.data
            np.grad = grad
            new_params.append(np)
        self.param_groups[0]["params"] = new_params

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