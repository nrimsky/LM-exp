import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

token = input("Enter your HF token: ")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed, norm):
        super().__init__()
        self.block = block
        self.unembedded = None
        self.unembed = unembed
        self.norm = norm

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.unembedded = self.unembed(self.norm(output[0]))
        return output

for i in range(len(model.model.layers)):
  if model.model.layers[i].__class__.__name__ != 'BlockOutputWrapper':
    model.model.layers[i] = BlockOutputWrapper(model.model.layers[i], model.lm_head, model.model.norm)

model = model.to(device)

tokens = tokenizer.encode('The capital of Germany is', return_tensors="pt").to(device)
output = model(tokens)
individual_decoded_tokens = list(tokenizer.batch_decode(tokens[0].unsqueeze(-1)))
for i, layer in enumerate(model.model.layers):
  values, indices = torch.topk(layer.unembedded[0], 10)
  print(f'Layer {i}')
  for i, token_decoded in enumerate(indices):
    print('    ', individual_decoded_tokens[i], '->', list(tokenizer.batch_decode(token_decoded.unsqueeze(-1))))