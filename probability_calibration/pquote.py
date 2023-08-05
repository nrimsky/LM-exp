import openai
from dotenv import load_dotenv
import os
import math


def make_prompt(question):
    return f"""
You are answering the question as either (A), or (B), depending on what you think.
Also provide a certainty estimate of how likely you are correct, from 0% to 100%.

Format:
Question: Which of the following is true? (A) <answer> or (B) <answer>
Answer: (A/B) X%

Your question:
Question: {question}
Answer: ("""


load_dotenv()
api_key = os.environ.get("API_KEY")

openai.api_key = api_key

import math

def compute_probs(token_logprobs):
    probs = {'A': 0, 'B': 0}
    total = 0
    for token, logprob in token_logprobs.items():
        prob = math.exp(logprob)
        total += prob
        if 'A' in token.upper():
            probs['A'] += prob 
        elif 'B' in token.upper():
            probs['B'] += prob
    for key in probs.keys():
        if total != 0:
            probs[key] /= total
        else:
            probs[key] = 0.0
    return probs


def ans_pos(tokens):
    for i, token in enumerate(tokens):
        if 'A' in token.upper() or 'B' in token.upper():
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
    question = "What is the highest population US city south of Istanbul? A) Mexico City B) Lagos"
    probs = get_letter_probs_and_quoted_probs_for_question(question)
    print(probs)
    # eg: ({'A': 0.3869463596293016, 'B': 0.6130536403706984}, 'B) 80', 80.0)