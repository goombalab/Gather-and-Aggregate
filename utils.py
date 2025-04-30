import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_eval(model, tokenizer, tasks, num_fewshot=0, batch_size=32, limit=None):
    results = simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", batch_size=batch_size),
        limit=limit,
        tasks=tasks,
        num_fewshot=num_fewshot,
        device="cuda",
        log_samples=False,
        batch_size=batch_size,
        verbosity="ERROR",
        cache_requests=True,
    )
    return results


def get_model(model_name, is_minimal=False):
    torch.cuda.empty_cache()
    
    if model_name == 'llama':
        # from models.modeling_llama import LlamaForCausalLM
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
        model.config.use_cache = False
        num_heads, head_dim = 32, 128

        # Alias the layers to match the Mamba naming scheme
        model.backbone = model.model
        for layer in model.backbone.layers:
            layer.layer_idx = layer.self_attn.layer_idx
            layer.mixer = layer.self_attn
            layer.mixer.out_proj = layer.mixer.o_proj
            
    elif model_name == 'falcon':
        model = AutoModelForCausalLM.from_pretrained('tiiuae/falcon-mamba-7b-instruct')
        tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-mamba-7b-instruct')
        num_heads, head_dim = 2 * model.config.hidden_size, 1

    elif model_name == 'llamba':
        # from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
        from models.llamba import LlambaLMHeadModel
        model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-8B-untied", strict=True)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        num_heads, head_dim = 32, 128

    elif model_name == 'zamba':
        # from transformers_zamba2 import AutoTokenizer as AutoTokenizerZamba2
        from models.modeling_zamba2 import Zamba2ForCausalLM
        model = Zamba2ForCausalLM.from_pretrained("Zyphra/Zamba2-7B", _attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-7B")
        num_heads, head_dim = 32, 224

        # Alias the layers to match the Mamba naming scheme
        model.backbone = model.model

    elif model_name == "unaligned_llamba":
        from models.llamba import LlambaLMHeadModel
        model = LlambaLMHeadModel.from_pretrained("goombalab/Llamba-8B-untied-unaligned", strict=True)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        num_heads, head_dim = 32, 128

    else:
        raise ValueError(f"Unknown model {model_name}")
    
    model = model.to(torch.bfloat16).to('cuda').eval()
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    return model, tokenizer, num_heads, head_dim

def get_minimal_model(model_name, layer_idx):
    model, tokenizer, num_heads, head_dim = get_model(model_name)
    model.backbone.layers = model.backbone.layers[0:layer_idx+1]
    return model, tokenizer, num_heads, head_dim

def keep_heads(model, layer_idx, heads, num_heads, head_dim):
    """
    Keep only the specified heads in the output projection of the specified layer.
    Zeroing the ouput projection of the other heads essentially removes their contribution to the output.
    """
    # Extract the weights of the layer's output projection
    weights = model.backbone.layers[layer_idx].mixer.out_proj.weight.data
    weights_view = weights.view(-1, num_heads, head_dim)

    # Create a mask to keep the specified heads
    mask = torch.zeros_like(weights_view)
    mask[:, heads, :] = 1.0

    # Apply the mask (in-place)
    weights_view *= mask


def remove_heads(model, layer_idx, num_heads, head_dim, heads):
    """
    Remove the specified heads from the output projection of the specified layer.
    Zeroing the ouput projection of the specified heads essentially removes their contribution to the output.
    """
    # Extract the weights of the layer's output projection
    weights = model.backbone.layers[layer_idx].mixer.out_proj.weight.data
    weights_view = weights.view(-1, num_heads, head_dim)

    # Create a mask to remove the specified heads
    mask = torch.ones_like(weights_view)
    mask[:, heads, :] = 0.0

    # Apply the mask (in-place)
    weights_view *= mask
