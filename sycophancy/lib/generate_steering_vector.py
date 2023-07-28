from datetime import datetime
from glob import glob
from random import sample
import os
import shutil
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def make_filenames(layer, use_timestamp=False):
    """
    layer: the layer index
    use_timestamp: whether to use a timestamp in the filenames
    """
    now = datetime.now()
    if use_timestamp:
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        return (
            f"./all_diffs_{timestamp}_layer_{layer}.pt",
            f"./vec_{timestamp}_layer_{layer}.pt",
        )
    return f"./all_diffs_layer_{layer}.pt", f"./vec_layer_{layer}.pt"


def generate_steering_vectors(llama_7b_helper, dataset, layers):
    """
    llama_7b_helper: a llama_7b_helper.Llama7bHelper object
    dataset: a dataset of (s_tokens, n_tokens) pairs
    layers: a list of layer indices to generate steering vectors for
    """
    filenames = dict([(layer, make_filenames(layer)) for layer in layers])
    diffs = dict([(layer, []) for layer in layers])

    for s_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        s_out = llama_7b_helper.get_logits(s_tokens.unsqueeze(0))
        for layer in layers:
            s_activations = llama_7b_helper.get_last_activations(layer)
            s_activations = s_activations[0, -2, :].detach().cpu()
            diffs[layer].append(s_activations)
        n_out = llama_7b_helper.get_logits(n_tokens.unsqueeze(0))
        for layer in layers:
            n_activations = llama_7b_helper.get_last_activations(layer)
            n_activations = n_activations[0, -2, :].detach().cpu()
            diffs[layer][-1] -= n_activations

    for layer in layers:
        diffs[layer] = torch.stack(diffs[layer])
        all_diffs_filename, vec_filename = filenames[layer]
        torch.save(diffs[layer], all_diffs_filename)
        vec = diffs[layer].mean(dim=0)
        unit_vec = vec / torch.norm(vec, p=2)
        torch.save(unit_vec, vec_filename)

    return filenames
