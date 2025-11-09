from typing import Dict, List, Tuple, Union, Optional
import warnings
import re
import torch
import torch.nn as nn
# from peft import (
#     PeftModel,
#     BaseTunerLayer,
# )
from .utils import get_nested_module, get_parameter_device, get_parameter_dtype
import logging

LOG = logging.getLogger(__name__)

def replace_lora_layers(model: nn.Module, attention_types: List[str]) -> dict:
    """
    Register the lora layers with batched weights and enabling gradient propagation.
    New attributes will be added to the LoraLayers, including lora_A_tensor and lora_B_tensor

    Args:
        model (nn.Module): the base model from PeftModel to be modified.

        attention_types(List[str]): the attention types to be replaced.
        
    Returns:
        dict: the dict contained pointer used to apply modification.
        
    """

    def register_lora_layer_forward(layer):
        def ca_forward(self):
            def forward(x, *args, **kwargs) -> torch.Tensor:
                self._check_forward_args(x, *args, **kwargs)

                # make sure the tensors are in the same batch position
                assert self.lora_A_tensor.shape[0] == self.lora_B_tensor.shape[0]
                assert self.lora_A_tensor.shape[0] == x.shape[0]
                adapter_names = kwargs.pop("adapter_names", None)
                assert adapter_names is None
                assert self.merged is False

                result = self.base_layer(x, *args, **kwargs)
                torch_result_dtype = result.dtype
                assert len(self.active_adapters) == 1
                lora_A, lora_B = self.lora_A_tensor, self.lora_B_tensor
                dropout = self.lora_dropout[self.active_adapters[0]]
                scaling = self.scaling[self.active_adapters[0]]
                x = x.to(lora_A.dtype)

                # The biggest modification
                result = (
                    result + torch.bmm(torch.bmm(dropout(x), lora_A), lora_B) * scaling
                ) 
                # try using vmap instead
                result = result.to(torch_result_dtype)

                return result

            return forward

        layer.forward = ca_forward(layer)
    
    model_lora_layers = {}
    
    # modification 1.
    # for name in model.base_model.targeted_module_names:
    for name in model.targeted_module_names:
        
        # modification 2.
        # if "q_proj" in name or "v_proj" in name:
        if name.split(".")[-2] in attention_types:

            # modification 3.
            # layer_to_replace = get_nested_module(model.base_model.model, name)
            layer_to_replace = eval(model+"."+name)
            layer_to_replace.lora_A_tensor = layer_to_replace.lora_A[model.active_adapter].weight.T.unsqueeze(0).clone()
            layer_to_replace.lora_B_tensor = layer_to_replace.lora_B[model.active_adapter].weight.T.unsqueeze(0).clone()
            
            # modification 4.
            # del layer_to_replace.lora_A[model.active_adapter] # torch compile requires every tensor used in forwarding
            # del layer_to_replace.lora_B[model.active_adapter]
            register_lora_layer_forward(layer_to_replace)
            model_lora_layers[name] = layer_to_replace  # need test
        else:
            raise ValueError(f"{name} is not a suported module")

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