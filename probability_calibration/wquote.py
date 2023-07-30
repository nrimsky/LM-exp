import openai
from dotenv import load_dotenv
import os
import math


def make_prompt(question):
    return f"""
You are an expert superforecaster that is very good at predicting the future and unbiased towards futuristic sounding things.
You are answering the question as either (A), or (B), depending on what you think.
Also provide a certainty estimate of how likely you are correct, using the keywords ["IMPOSSIBLE", "UNLIKELY", "EVEN", "LIKELY", "CERTAIN"].

Format:
Question: Which of the following is true? (A) <answer> or (B) <answer>
Answer: (A/B) IMPOSSIBLE/UNLIKELY/EVEN/LIKELY/CERTAIN

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

def get_weighted_average_quoted_prob(top_logprobs):
    probs = {'IMPOSSIBLE': 0, 'UNLIKELY': 0, 'EVEN': 0, 'LIKELY': 0, 'CERTAIN': 0}
    total = 0
    for token, logprob in top_logprobs.items():
        prob = math.exp(logprob)
        first_letter = token.strip().upper()[0]
        total += prob
        if 'I' == first_letter:
            probs['IMPOSSIBLE'] += prob 
        elif 'U' == first_letter:
            probs['UNLIKELY'] += prob
        elif 'E' == first_letter:
            probs['EVEN'] += prob
        elif 'L' == first_letter:
            probs['LIKELY'] += prob
        elif 'C' == first_letter:
            probs['CERTAIN'] += prob
    for key in probs.keys():
        if total != 0:
            probs[key] /= total
        else:
            probs[key] = 0.0
    represented_values = {'IMPOSSIBLE': 0, 'UNLIKELY': 25, 'EVEN': 50, 'LIKELY': 75, 'CERTAIN': 100}
    weighted_average = 0
    for key in probs.keys():
        weighted_average += probs[key] * represented_values[key]
    return weighted_average
    

def get_letter_probs(response):
    answer = response["choices"][0]["logprobs"]["tokens"]
    text_answer = response["choices"][0]["text"]
    pos = ans_pos(answer)
    top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][pos]
    probs = compute_probs(top_logprobs)
    return probs, text_answer, get_weighted_average_quoted_prob(response["choices"][0]["logprobs"]["top_logprobs"][pos + 2])

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
    question = "Which of the following is true? (A) Next year we will have more AI agents. (B) Next year we will have fewer AI agents."
    probs = get_letter_probs_and_quoted_probs_for_question(question)
    print(probs)
    # e.g. ({'A': 0.9329304942712406, 'B': 0.06706950572875926}, 'A) LIKELY', 75.00023384320643)