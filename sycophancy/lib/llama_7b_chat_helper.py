"""
Wrapper class for Llama2 7B Chat model 
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


class Llama7BChatHelper:
    def __init__(
        self, token, system_prompt="You are a helpful, honest and concise assistant."
    ):
        """
        token: huggingface token for Llama-2-13b-chat-hf
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=token
        ).to(self.device)
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer
            )
    
    def prompt_to_tokens(self, instruction):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        dialog_content = B_SYS + self.system_prompt + E_SYS + instruction.strip()
        dialog_tokens = self.tokenizer.encode(
            f"{B_INST} {dialog_content.strip()} {E_INST}"
        )
        return torch.tensor(dialog_tokens).unsqueeze(0)
    
    def generate_text(self, prompt, max_length=100):
        tokens = self.prompt_to_tokens(prompt).to(self.device)
        generated = self.model.generate(
            inputs=tokens.to(self.device),
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(generated)[0]

    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].last_hidden_state

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()
