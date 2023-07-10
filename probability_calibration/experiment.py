import json
from get_ans import get_letter_probs_for_question
from pquote import get_letter_probs_and_quoted_probs_for_question
from wquote import get_letter_probs_and_quoted_probs_for_question_words
from time import sleep

def get_prompts(dataset_path, results_path):
    dataset = None
    with open(dataset_path) as dfile:
        dataset = json.load(dfile)
    results = []
    for item in dataset:
        for _ in range(3):
            try:
                probs_just_yn = get_letter_probs_for_question(item["question"])
                probs, text_answer, probability_said = get_letter_probs_and_quoted_probs_for_question(item["question"])
                probs_words, text_answer_words, probability_said_words = get_letter_probs_and_quoted_probs_for_question_words(item["question"])
                item["logit_probs_full_q_words"] = probs_words
                item["text_answer_full_q_words"] = text_answer_words
                item["probability_said_full_q_words"] = probability_said_words
                item["logit_probs_just_yn"] = probs_just_yn
                item["logit_probs_full_q"] = probs
                item["text_answer_full_q"] = text_answer
                item["probability_said_full_q"] = probability_said
                results.append(item)
                break
            except:
                # rate limiting
                sleep(15)
        
    with open(results_path, 'w') as rfile:
        json.dump(results, rfile)
    return results

if __name__ == "__main__":
    get_prompts("questions.json", "results.json")