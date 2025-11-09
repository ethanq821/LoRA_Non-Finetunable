import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from typing import List

from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, PretrainedConfig
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import cast_training_params, _set_state_dict_into_text_encoder
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import convert_unet_state_dict_to_peft, convert_state_dict_to_diffusers, get_adapter_name
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer

from peft import LoraConfig
from functools import partial
from meta.lora_maml import replace_lora_layers#, set_loras




def unwrap_model(model,accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
def save_model_hook(models, weights, output_dir, accelerator, unet, text_encoder):
    if accelerator.is_main_process:
        # there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder atten layers
        unet_lora_layers_to_save = None
        text_encoder_lora_layers_to_save = None

        for model in models:
            if isinstance(model, type(unwrap_model(unet, accelerator))):
                unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            elif isinstance(model, type(unwrap_model(text_encoder, accelerator))):
                text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)
                )
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        StableDiffusionLoraLoaderMixin.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_lora_layers_to_save,
        )

def load_model_hook(models, input_dir, train_text_encoder, mixed_precision, unet, text_encoder, logger, accelerator):
    unet_ = None
    text_encoder_ = None

    while len(models) > 0:
        model = models.pop()

        if isinstance(model, type(unwrap_model(unet, accelerator))):
            unet_ = model
        elif isinstance(model, type(unwrap_model(text_encoder, accelerator))):
            text_encoder_ = model
        else:
            raise ValueError(f"unexpected save model: {model.__class__}")

    lora_state_dict, network_alphas = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)

    unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
    unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
    incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            logger.warning(
                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                f" {unexpected_keys}. "
            )

    if train_text_encoder:
        _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_)

    # Make sure the trainable params are in float32. This is again needed since the base models
    # are in `weight_dtype`. More details:
    # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
    if mixed_precision == "fp16":
        models = [unet_]
        if train_text_encoder:
            models.append(text_encoder_)

        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

def load_base_model(
    pretrained_model_name_or_path:str=None,
    revision:str=None,
    variant:str=None,
):
    # from dreambooth training script
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        text_encoder_cls = CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        text_encoder_cls = RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        text_encoder_cls = T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")
    
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision, use_fast=False
    )
    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=revision, variant=variant
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    print(f"Loading model from {pretrained_model_name_or_path} with revision {revision}")

    return noise_scheduler, tokenizer, text_encoder, vae, unet


def prepare_model_for_training(
    is_normal,
    text_encoder = None, 
    vae = None, 
    unet = None,
    accelerator: Accelerator=None,
    mixed_precision:str=None,
    gradient_checkpointing:bool=True,
    train_text_encoder:bool=False,
    rank:int=8,
    lora_alpha:int=16,
    lora_dropout:float=0.0,
    init_lora_weights:str = "gaussian",
    unet_target_modules:List[str]=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    text_encoder_target_modules:List[str]=["q_proj", "k_proj", "v_proj", "out_proj"],
    logger = None,
):
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    all_lora_layers = []
    unet_lora_layers = create_or_replace_lora(is_normal, unet, rank, rank, lora_dropout, 'gaussian', unet_target_modules)
    all_lora_layers.append(unet_lora_layers) if unet_lora_layers else ...
    if train_text_encoder:
        text_encoder_lora_layers = create_or_replace_lora(is_normal, text_encoder, rank, rank, lora_dropout, 'gaussian', text_encoder_target_modules)
        all_lora_layers.append(text_encoder_lora_layers) if text_encoder_lora_layers else ...
        
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    load_hook = partial(load_model_hook, train_text_encoder=train_text_encoder, unet=unet, text_encoder=text_encoder, logger=logger, accelerator=accelerator,)
    save_hook = partial(save_model_hook, accelerator=accelerator, unet=unet, text_encoder=text_encoder,)
    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    # Make sure the trainable params are in float32.
    if mixed_precision == "fp16":
        models = [unet]
        if train_text_encoder:
            models.append(text_encoder)

        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
    
    return text_encoder, vae, unet, weight_dtype, all_lora_layers

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
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_lora_weights=init_lora_weights,
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

def save_model():
    ...

if __name__ == "__main__":
    # 1. 测试create_or_replace_lora中的replace_lora_layers函数
    #    create_or_replace_lora调用了replace_lora_layers函数来得到一个类似c语言的指针，以字典形式保存为lora_layers。这样做是为了减少maml运算时重复递归访问模型。
    #
    #    测试任务：1.1 测试字典形式的指针是否真的可以指向模型中的lora层，还是只是保存了lora层的名字和参数。可以通过修改一下字典lora_layers中的参数，看看修改前后是否会对**同一组输入**产生不同的输出进行。
    #            1.2 测试replace_lora_layers能否正确捕获lora。能正确得到输出结果即可。
    #    
    #    提示：
    #        1.1的相关模型的加载代码和**dummy_input**已经写好了。
    #        同时，可以在注释里写一下replace_lora_layers反馈的任意一个键和任意一个值的形状，方便未来理解。
    
    ## prepare the model
    from diffusers import StableDiffusionPipeline
    # from accelerate.utils import set_seed
    # set_seed(42)
    pipe = StableDiffusionPipeline.from_pretrained("/data1/dif/sd1-5", torch_dtype=torch.bfloat16).to("cpu")
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    noise_scheduler = pipe.scheduler

    ## run the unet
    unet_lora_layers_1 = create_or_replace_lora(
        is_normal=False, 
        model=unet, 
        r=8, 
        lora_alpha=8, 
        lora_dropout=0, 
        init_lora_weights= 'gaussian', 
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"]
    )

    dummy_input = torch.randn(1, 4, 64, 64).to("cpu", dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 77, 768).to("cpu", dtype=torch.bfloat16)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device="cpu")
    timesteps = timesteps.long()
    
    model_pred_0 = unet(dummy_input, timesteps, encoder_hidden_states, return_dict=False)[0]
    model_pred_1 = unet(dummy_input, timesteps, encoder_hidden_states, return_dict=False)[0]

    ## modify the unet through unet_lora_layers and then run it again
    for k, v in unet_lora_layers_1.items():
        v.data += torch.randn(v.shape).to("cpu")

    model_pred_2 = unet(dummy_input, timesteps, encoder_hidden_states, return_dict=False)[0]
    assert torch.allclose(model_pred_0, model_pred_1, atol=1e-5)
    assert not torch.allclose(model_pred_0, model_pred_2, atol=1e-5)

    # 2. 测试create_or_replace_lora能否加载多个lora。
    #
    #    在以后的大实验中需要在一个模型中重复加载和卸载lora来对不同的人物数据集进行微调，但是不知道是否能够正确切换lora。
    #    测试任务：测试lora是否能够正确切换。你需要用create_or_replace_lora加载一个lora，然后用create_or_replace_lora再加载另一个lora，比较两个lora的输出是否相同。
    #
    #    提示：create_or_replace_lora本身包含了许多调用有用的方法，可以先从它们的库开始看。