import openai
from dotenv import load_dotenv
import os
import math
from random import randint


def make_prompt(question):
    # randomize example probabilities to avoid biasing to those numbers
    r1 = randint(75, 95)
    r2 = randint(55, 75)
    r3 = randint(20, 40)
    return f"""
You are an expert superforecaster that is very good at predicting the future and unbiased towards futuristic sounding things.
You are answering the question as either Yes (Y), or No (N), depending on what you think.
Also provide a probability estimate that the answer is Yes, from 0% to 100%.

Examples:
Question: Will self-driving cars become the predominant mode of transportation by 2040?
Answer: (Y) {r1}%

Question: Will humans achieve interstellar travel within the next 500 years?
Answer: (Y) {r2}%

Question: Will the world's population be over 12 billion by 2050?
Answer: (N) {r3}%

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
    text_answer = response["choices"][0]["text"]
    probability_said = float(text_answer.split(" ")[-1])
    pos = ans_pos(answer)
    top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][pos]
    probs = compute_probs(top_logprobs)
    return probs, text_answer, probability_said

def get_letter_probs_and_quoted_probs_for_question(question):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=make_prompt(question),
        max_tokens=5,
        temperature=0,
        logprobs=5,
        stop=['%']
    )
    return get_letter_probs(response)


if __name__ == "__main__":
    question = "Did the coin come up heads or tails?"
    probs = get_letter_probs_and_quoted_probs_for_question(question)
    print(probs)