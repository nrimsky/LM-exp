import openai
from dotenv import load_dotenv
import os
import json
import os

load_dotenv()
api_key = os.environ.get("API_KEY")

openai.api_key = api_key

PROMPT = """You are an assistant that generates a dataset, formatted as a JSON list. The structure of the dataset should be like:
[
{"bio": "<some description of a person>"},
{"bio": "<some description of a person>"},
...
]
Please include at least 20 entries in the dataset. *Do not* include any additional text besides the dataset itself."""

def get_dataset(dataset_description, save_to):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": PROMPT,
            },
            {"role": "user", "content": dataset_description},
        ],
    )
    statements = response["choices"][0]["message"]["content"].strip()
    try:
        statements = json.loads(statements)
        with open(save_to, "w") as sfile:
            json.dump(statements, sfile)
    except json.decoder.JSONDecodeError:
        print("Error: dataset not formatted as a JSON list")
        with open(save_to, "w") as sfile:
            sfile.write(statements)

if __name__ == "__main__":
    get_dataset(f"Generate a dataset of 30 short bios of randomly sampled Americans.", "datasets/american_bios.json")
