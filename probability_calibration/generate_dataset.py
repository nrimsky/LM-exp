import openai
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.environ.get("API_KEY")

openai.api_key = api_key

PROMPT = """You are an assistant that generates a dataset of Yes/No future forecasting questions, formatted in the following way:
[
{"question": "Will the use of traditional paper books become rare by 2040?"},
{"question": "Will advancements in medical technology double the average human lifespan by 2100?"},
{"question": "Will global temperatures rise by more than 3 degrees Celsius by 2100?"},
...
]
"""


def get_dataset(save_to):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "user",
                "content": "Please generate a dataset of at least 50 questions that require attempting to forecast, on topics such as media, culture, fashion, tech, AI, medicine, celebrities, names of things, world records, sports, animals, art. Both Yes and No should be somewhat likely - the answer should not be clear cut. Make the questions varied, niche, and on a variety of topics. Do not include any additional text besides the dataset.",
            },
        ],
        temperature=1.3
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
    get_dataset("questions4.json")
    print(f"Saved dataset to questions4.json")
