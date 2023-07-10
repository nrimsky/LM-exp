import openai
from dotenv import load_dotenv
import os
import math


def make_prompt(question):
    return f"""
You are an expert superforecaster that is very good at predicting the future and unbiased towards futuristic sounding things.
You are answering the question as either Yes (Y), or No (N), depending on what you think.

Examples:
Question: Will self-driving cars become the predominant mode of transportation by 2040?
Answer: (Y)

Question: Will humans achieve interstellar travel within the next 500 years?
Answer: (Y)

Question: Will the world's population be over 12 billion by 2050?
Answer: (N)

Your question:
Question: {question}
Answer: ("""


load_dotenv()
api_key = os.environ.get("API_KEY")

openai.api_key = api_key

import math

def compute_probs(token_logprobs):
    probs = {'Y': 0, 'N': 0}
    total = 0
    for token, logprob in token_logprobs.items():
        prob = math.exp(logprob)
        total += prob
        if 'Y' in token.upper():
            probs['Y'] += prob 
        elif 'N' in token.upper():
            probs['N'] += prob
    for key in probs.keys():
        if total != 0:
            probs[key] /= total
        else:
            probs[key] = 0.0
    return probs


def ans_pos(tokens):
    for i, token in enumerate(tokens):
        if 'Y' in token.upper() or 'N' in token.upper():
            return i
    return -1
    

def get_letter_probs(response):
    answer = response["choices"][0]["logprobs"]["tokens"]
    pos = ans_pos(answer)
    top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][pos]
    probs = compute_probs(top_logprobs)
    return probs

def get_letter_probs_for_question(question):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=make_prompt(question),
        max_tokens=5,
        temperature=0,
        logprobs=5,
        stop=[')']
    )
    return get_letter_probs(response)


if __name__ == "__main__":
    question = "Will the majority of the world's energy come from renewable sources by 2060?"
    probs = get_letter_probs_for_question(question)
    print(probs)