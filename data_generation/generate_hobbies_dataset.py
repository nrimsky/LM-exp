import openai
from dotenv import load_dotenv
import os
import json
from hobbies import HOBBIES
import os
from time import sleep

load_dotenv()
api_key = os.environ.get("API_KEY")

openai.api_key = api_key

PROMPT = """You are an assistant that generates a dataset, formatted as a JSON list. The structure of the dataset should be like:
[
{"input_text": "Discovering a hidden treasure chest filled with ancient artifacts while geocaching.", "target_text": "The person would likely feel ecstatic, thrilled, and amazed at the unexpected and valuable discovery."},
{"input_text": "A surfer encounters a shark attack.", "target_text": "Intense fear and panic because of the danger of the situation and the possibility of being eaten."},
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
    dir = os.path.join("datasets", "hobbies")
    for x in HOBBIES:
        filename = os.path.join(dir, f'{"_".join(x.split(" "))}.json')
        # check if file exists
        if os.path.exists(filename):
            continue
        for _ in range(3):
            try:
                get_dataset(f"Generate a dataset that pairs brief, clear descriptions of potential events or occurrences related to the hobby of {x}, with a brief description of how someone is likely to feel in response to these events.", filename)
                print(f"Saved dataset to {filename}")
                break
            except openai.error.ServiceUnavailableError:
                # rate limit exceeded
                print("Rate limit exceeded, waiting 20 seconds")
                sleep(20)
