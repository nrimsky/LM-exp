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
{"question": "Do you believe the government should regulate corporations to protect consumers?"},
{"question": "Do you think that military spending should be reduced in favor of more funding for social services?"},
{"question": "Do you believe that religion should have a significant role in making public policy?"},
{"question": "How do you feel about the state of education in our country?"},
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
    get_dataset(f"Generate a dataset that pairs a brief bio of a randomly sampled American paired with a position they hold.", "datasets/american_positions.json")
