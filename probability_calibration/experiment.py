import json
from get_ans import get_ans
from time import sleep

def get_prompts(dataset_path, results_path, model="text-davinci-003"):
    dataset = None
    with open(dataset_path) as dfile:
        dataset = json.load(dfile)
    results = []
    for item in dataset:
        for _ in range(10):
            try:
                logit_probs_base = get_ans(item["question"], "just_yes_no", model=model)
                logit_probs_numerical_likelihood, text_answer_numerical_likelihood, probability_quoted_numerically = get_ans(item["question"], "yes_no_numerical_likelihood", model=model)
                logit_probs_word_likelihood, text_answer_word_likelihood, probability_quoted_verbally = get_ans(item["question"], "yes_no_word_likelihood", model=model)
                item["logit_probs_base"] = logit_probs_base
                item["logit_probs_numerical_likelihood"] = logit_probs_numerical_likelihood
                item["text_answer_numerical_likelihood"] = text_answer_numerical_likelihood
                item["probability_quoted_numerically"] = probability_quoted_numerically
                item["logit_probs_word_likelihood"] = logit_probs_word_likelihood
                item["text_answer_word_likelihood"] = text_answer_word_likelihood
                item["probability_quoted_verbally"] = probability_quoted_verbally
                results.append(item)
                break
            except:
                # Handle rate limiting errors
                sleep(15)
    with open(results_path, 'w') as rfile:
        json.dump(results, rfile)
    return results

if __name__ == "__main__":
    get_prompts("questions.json", "davinci.json", model="davinci")
    get_prompts("questions.json", "text_davinci_3.json", model="text-davinci-003")