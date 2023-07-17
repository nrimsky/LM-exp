import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.last_hidden_state = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.last_hidden_state = output[0]
        return output

class Llama7BHelper:
    def __init__(self, save_layer_idx, pretrained_model="huggyllama/llama-7b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model).to(self.device)
        self.save_layer_idx = save_layer_idx
        self.model.model.layers[save_layer_idx] = BlockOutputWrapper(self.model.model.layers[save_layer_idx])

    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
          logits = self.model(inputs.input_ids.to(self.device)).logits
          return logits

    def get_last_activations(self):
        return self.model.model.layers[self.save_layer_idx].last_hidden_state
    
    def get_yes_log_odds(self, model_input):
        yes_token = int(self.tokenizer("yes", return_tensors="pt").input_ids[0][1])
        no_token = int(self.tokenizer("no", return_tensors="pt").input_ids[0][1])
        inputs = self.tokenizer(model_input, return_tensors="pt")
        with torch.no_grad():
          logits = self.model(inputs.input_ids.to(self.device)).logits
        final_token_logits = logits[0][-1][:]
        return final_token_logits[yes_token] - final_token_logits[no_token]

def get_activations(model, prompt, token_idx):
    model.get_logits(prompt['bio'])
    activations = model.get_last_activations() # batch size x n tokens x dict size
    return activations[0, token_idx, :]

def combine_prompt_and_question(prompt, question):
    return f"{prompt['bio']} When asked the question '{question}', yes or no, {prompt['pronoun']} answered: "

def get_yes_log_odds(model, model_input):
    return model.get_yes_log_odds(model_input)

def get_question_answer_for_prompt(model, question, prompt):
    model_input = combine_prompt_and_question(prompt, question)
    yes_log_odds = get_yes_log_odds(model, model_input)
    return float(yes_log_odds)

def get_question_answers_for_prompt(model, questions, prompt):
    yes_lo = []
    for question in questions:
        yes_lo.append(get_question_answer_for_prompt(model, question, prompt))
    return torch.tensor(yes_lo)
    
def get_q_vectors_for_prompts(model, questions, prompts, token_idx, layer):
    q_vectors = []
    activations = []
    for i, prompt in enumerate(prompts):
        prompt_activations = get_activations(model, prompt, token_idx)
        qas = get_question_answers_for_prompt(model, questions, prompt)
        tokenized_prompt = model.tokenizer(prompt['bio'], return_tensors="pt").input_ids
        torch.save({
            "prompt_activations": prompt_activations,
            "question_answers": qas,
            "tokenized_prompt": tokenized_prompt
        },  f"drive/MyDrive/steering_data/llama7b_{token_idx}_{layer}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        q_vectors.append(qas)
        activations.append(prompt_activations)
    return torch.stack(q_vectors), torch.stack(activations)

def collect_data(model, questions, prompts, token_idx, layer):
    q_vectors, activations = get_q_vectors_for_prompts(model, questions, prompts, token_idx, layer)
    return q_vectors, activations