import openai
from dotenv import load_dotenv
import os
import json
from cuisines import FOOD_CATEGORIES
import os
from time import sleep

load_dotenv()
api_key = os.environ.get("API_KEY")

openai.api_key = api_key

PROMPT = """You are an assistant that generates a dataset, formatted as a JSON list. The structure of the dataset should be like:
[
{"input_text": "Chicken Pie", "target_text": "chicken, vegetables, flour, butter, milk, salt, pepper, bay leaf"},
{"input_text": "Vanilla Cake", "target_text": "sugar, flour, butter, eggs, vanilla extract, baking powder, milk"},
...
]
Please include at least 20 entries in the dataset. *Do not* include any additional text besides the dataset itself."""

def get_dataset(dataset_description, save_to):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
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
    dir = os.path.join("datasets", "cooking")

    filename = os.path.join(dir, 'rich_recipes.json')
    get_dataset(f"Generate a dataset mapping rich, buttery and fatty recipe names to comma seperated ingredients with spaces. Recipe names and ingredients should be in English (for a US audience).", filename)

    # for x in FOOD_CATEGORIES:
    #     filename = os.path.join(dir, f'{"_".join(x.split(" "))}_recipes.json')
    #     # check if file exists
    #     if os.path.exists(filename):
    #         continue
    #     for _ in range(3):
    #         try:
    #             get_dataset(f"Generate a dataset mapping {x} recipe names to comma seperated ingredients with spaces. Recipe names and ingredients should be in English (for a US audience).", filename)
    #             print(f"Saved dataset to {filename}")
    #             break
    #         except openai.error.ServiceUnavailableError:
    #             # rate limit exceeded
    #             print("Rate limit exceeded, waiting 20 seconds")
    #             sleep(20)
