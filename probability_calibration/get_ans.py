import openai
from dotenv import load_dotenv
import os
from helpers import compute_probs, ans_pos

load_dotenv()
api_key = os.environ.get("API_KEY")

openai.api_key = api_key

def make_prompt_yn(question):
    return f"""
Final round of superforecaster questions:
Congratulations on making it to the final round! You are answering the question as either Yes (Y), or No (N), depending on what you think.

Answer format example:
Answer: (N)

Your question: {question}
Answer: ("""

def make_prompt_wp(question):
    return f"""
Final round of superforecaster questions:
Congratulations on making it to the final round! You are answering the question as either Yes (Y), or No (N), depending on what you think.
Also estimate how likely the answer is Yes - select one of Improbable, Unlikely, Uncertain, Likely, or Certain.

Answer format example:
Answer: (N) Improbable

Your question: {question}
Answer: ("""

def make_prompt_np(question):
    return f"""
Final round of superforecaster questions:
Congratulations on making it to the final round! You are answering the question as either Yes (Y), or No (N), depending on what you think.
Also provide a probability estimate of Yes, from 0% to 100%.

Answer format example:
Answer: (N) 7%

Your question: {question}
Answer: ("""

def get_letter_probs_yn(response):
    answer = response["choices"][0]["logprobs"]["tokens"]
    pos = ans_pos(answer)
    top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][pos]
    probs = compute_probs(top_logprobs)
    return probs

def get_letter_probs_wp(response):
    answer = response["choices"][0]["logprobs"]["tokens"]
    text_answer = response["choices"][0]["text"]
    probability_said  = -1
    phrase = text_answer.split(" ")[-1].lower()
    if "improbable" in phrase:
        probability_said = 10
    elif "unlikely" in phrase:
        probability_said = 30
    elif "uncertain" in phrase:
        probability_said = 50
    elif "likely" in phrase:
        probability_said = 70
    elif "certain" in phrase:
        probability_said = 90
    pos = ans_pos(answer)
    top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][pos]
    probs = compute_probs(top_logprobs)
    return probs, text_answer, probability_said

def get_letter_probs_np(response):
    answer = response["choices"][0]["logprobs"]["tokens"]
    text_answer = response["choices"][0]["text"]
    probability_said = float(text_answer.split(" ")[-1])
    pos = ans_pos(answer)
    top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][pos]
    probs = compute_probs(top_logprobs)
    return probs, text_answer, probability_said

PROMPT_FACTORIES = {
    "just_yes_no": (make_prompt_yn, get_letter_probs_yn),
    "yes_no_word_likelihood": (make_prompt_wp, get_letter_probs_wp),
    "yes_no_numerical_likelihood": (make_prompt_np, get_letter_probs_np)
}

STOP = {
    "just_yes_no": [")"],
    "yes_no_word_likelihood": ["\n"],
    "yes_no_numerical_likelihood": ["%"]
}

def get_ans(question, prompt_type, model="text-davinci-003"):
    response = openai.Completion.create(
        model=model,
        prompt=PROMPT_FACTORIES[prompt_type][0](question),
        max_tokens=5,
        temperature=0,
        logprobs=5,
        stop=STOP[prompt_type]
    )
    return PROMPT_FACTORIES[prompt_type][1](response)


if __name__ == "__main__":
    question = "Will nuclear fusion be a viable energy source by 2030?"
    probs = get_ans(question, "yes_no_word_likelihood", model="text-davinci-003")
    print(probs)