import json
from random import sample
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch.nn.functional as F

DATASETS = [
    "sycophancy_on_nlp_survey.jsonl",
    "sycophancy_on_philpapers2020.jsonl",
    "sycophancy_on_political_typology_quiz.jsonl",
]


class ComparisonDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        s_text = item["sycophantic_text"]
        n_text = item["non_sycophantic_text"]

        s_tokens = self.tokenizer.encode(s_text, return_tensors="pt")[0]
        n_tokens = self.tokenizer.encode(n_text, return_tensors="pt")[0]

        return s_tokens, n_tokens


def get_json_data():
    all_data = []
    for item in DATASETS:
        url = f"https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/{item}"
        r = requests.get(url).text
        data = [json.loads(l) for l in r.split("\n") if l != ""]
        for d in data:
            all_data.append(
                {
                    "sycophantic_text": "Question: "
                    + d["question"]
                    + d["answer_matching_behavior"],
                    "non_sycophantic_text": "Question: "
                    + d["question"]
                    + d["answer_not_matching_behavior"],
                }
            )
    return all_data


def get_dataset(tokenizer, n_pairs=5000):
    data = get_json_data()
    return ComparisonDataset(sample(data, n_pairs), tokenizer)
