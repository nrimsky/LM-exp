"""
Wrapper class for Llama 7b model 
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.last_hidden_state = None
        self.add_activations = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.last_hidden_state = output[0]
        if self.add_activations is not None:
            output = (output[0] + self.add_activations,) + output[1:]
        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.last_hidden_state = None
        self.add_activations = None

    
class Llama27BHelper:
    def __init__(self, pretrained_model="meta-llama/Llama-2-7b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model, use_auth_token=token).to(self.device)
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens.to(self.device)).logits
            return logits
    
    def get_last_activations(self, layer):
        return self.model.model.layers[layer].last_hidden_state

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

