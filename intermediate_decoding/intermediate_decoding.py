import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None
        
    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        
        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm
        
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None
        self.attn_output_unembedded = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))
        mlp_output = self.block.mlp(self.post_attention_layernorm(self.block.self_attn.activations))
        self.mlp_output_unembedded = self.unembed_matrix(self.norm(mlp_output))
        attn_output = self.block.self_attn.activations
        self.attn_output_unembedded = self.unembed_matrix(self.norm(attn_output))
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()

    def get_attn_activations(self):
        return self.block.self_attn.activations

class Llama7BV2Helper:
    def __init__(self, token):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=token).to(self.device)
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
          logits = self.model(inputs.input_ids.to(self.device)).logits
          return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)
    
    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def decode_all_layers(self, text, topk=10, print_attn=True, print_mlp=True, print_block=True):
        """
        Print the top k tokens for each layer's attention, MLP, and block decoded outputs
        """
        self.get_logits(text)
        for i, layer in enumerate(self.model.model.layers):
            print(f'Layer {i}')
            if print_attn:
                values, indices = torch.topk(layer.attn_output_unembedded[0], topk)
                print(f'Attention output', list(self.tokenizer.batch_decode(indices[-1].unsqueeze(-1))))
            if print_mlp:
                values, indices = torch.topk(layer.mlp_output_unembedded[0], topk)
                print(f'MLP output', list(self.tokenizer.batch_decode(indices[-1].unsqueeze(-1))))
            if print_block:
                values, indices = torch.topk(layer.block_output_unembedded[0], topk)
                print(f'Block output', list(self.tokenizer.batch_decode(indices[-1].unsqueeze(-1))))