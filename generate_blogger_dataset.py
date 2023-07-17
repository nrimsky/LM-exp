import openai
from dotenv import load_dotenv
import os
import json
import os

load_dotenv()
api_key = os.environ.get("API_KEY")

openai.api_key = api_key

PROMPT = """You are an assistant that generates blogpost introductions, formatted as a JSON list. The structure of the dataset should be like:
[
{"bio": "Hi, my name is Tom and I am starting this blog to discuss my ideas and opinions. I am philosophy professor at the University of Cambridge studying morality."},
{"bio": "Hello, I'm Katherine and this is my personal blog about my worldview and perspectives! My day job is as a software engineer at Google. My passion is arts and crafts."},
...
]
Please include at least 30 entries in the dataset. *Do not* include any additional text besides the dataset itself."""

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
    get_dataset(f"Generate a dataset of 30 short diverse blogpost introductions.", "datasets/blog_intros.json")
