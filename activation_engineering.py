import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

def get_log_prob(logits, token_ids, start_idx):
    p = torch.nn.functional.softmax(logits[0], dim=-1)
    logprobs = torch.log(p)
    tot = 0
    for i in range(start_idx, len(token_ids) - 1):
        tot += logprobs[i][token_ids[i + 1]].item()
    return tot

def pad_tensors_to_same_size(tensor1, tensor2):
    # Ensure tensor2 is no larger than tensor1 along the second dimension
    if tensor2.size(1) > tensor1.size(1):
        tensor2 = tensor2[:, :tensor1.size(1), :]

    # In case tensor2 is smaller, pad it with zeros to match tensor1's size
    padding_size2 = max(0, tensor1.size(1) - tensor2.size(1))
    if padding_size2 > 0:
        padding2 = torch.zeros((tensor2.size(0), padding_size2, tensor2.size(2)), device=tensor2.device)
        tensor2 = torch.cat([tensor2, padding2], dim=1)

    return tensor1, tensor2

def prompt_completion_log_prob(p1, p2, model, tokenizer, device="cpu"):
    p1 = tokenizer(p1, return_tensors="pt").input_ids
    p2 = tokenizer(p2, return_tensors="pt").input_ids[:, 2:]
    p2_start_idx = p1.size(1)
    inputs = torch.cat([p1, p2], dim=1)
    with torch.no_grad():
        logits = model(inputs.to(device)).logits
    return get_log_prob(logits, inputs[0], p2_start_idx)

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
            o1, o2 = pad_tensors_to_same_size(output[0], self.add_activations)
            output = (o1 + o2,) + output[1:]
        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.last_hidden_state = None
        self.add_activations = None

class ValueProjWrapper(torch.nn.Module):
    def __init__(self, vproj):
        super().__init__()
        self.vproj = vproj
        self.last_values = None
        self.add_values = None

    def forward(self, *args, **kwargs):
        output = self.vproj(*args, **kwargs)
        self.last_values = output
        if self.add_values is not None:
            o1, o2 = pad_tensors_to_same_size(output, self.add_values)
            return o1 + o2
        return output

    def add(self, add_values):
        self.add_values = add_values

    def reset(self):
        self.last_values = None
        self.add_values = None
    
class Llama7BHelper:
    def __init__(self, pretrained_model="huggyllama/llama-7b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model).to(self.device)
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i].self_attn.v_proj = ValueProjWrapper(layer.self_attn.v_proj)
            self.model.model.layers[i] = BlockOutputWrapper(layer)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
          logits = self.model(inputs.input_ids.to(self.device)).logits
          return logits
    
    def get_last_activations(self, layer):
        return self.model.model.layers[layer].last_hidden_state

    def get_last_attn_values(self, layer):
        return self.model.model.layers[layer].block.self_attn.v_proj.last_values
    
    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def set_add_attn_values(self, layer, values):
        self.model.model.layers[layer].block.self_attn.v_proj.add(values)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.block.self_attn.v_proj.reset()
            layer.reset()

    def mix_activations(self, base_input, mixing_input, multiplier, layer, values_only=False, max_length=100):
        self.reset_all()
        self.get_logits(mixing_input)
        mixing_values = None
        if values_only:
            mixing_values = self.get_last_attn_values(layer)
        else:
            mixing_values = self.get_last_activations(layer)
        mixing_values *= multiplier
        if values_only:
            self.set_add_attn_values(layer, mixing_values)
        else:
            self.set_add_activations(layer, mixing_values)
        return self.generate_text(base_input, max_length=max_length)

    def activation_mixing_experiment(self, base_input, mixing_input, multipliers, layers, max_length=50):
        """
        base_input: The input to be modified
        mixing_input: The input to be mixed in
        multipliers: A list of multipliers to test appling to the mixing activations
        layers: A list of layers to test mixing activations from
        max_length: The maximum length of the generated text
        
        Returns: A CSV file for results with corresponding multipliers, layers, and whether full block activations or just attention values were modified
        """
        self.get_logits(mixing_input)
        results = []
        for layer in layers:
            self.reset_all()
            self.get_logits(mixing_input)
            mixing_values = self.get_last_attn_values(layer)
            mixing_activations = self.get_last_activations(layer)
            for multiplier in multipliers:
                self.reset_all()
                self.set_add_attn_values(layer, mixing_values * multiplier)
                v = self.generate_text(base_input, max_length=max_length)
                results.append((multiplier, layer, "attn", v))
                self.reset_all()
                self.set_add_activations(layer, mixing_activations * multiplier)
                a = self.generate_text(base_input, max_length=max_length)
                results.append((multiplier, layer, "activations", a))
        df = pd.DataFrame(results, columns=["multiplier", "layer", "type", "text"])
        df.to_csv("results.csv", index=False, sep=',', escapechar='\\')
        return df
        



