import torch
import json
from tqdm import tqdm


def eval_truthful_qa(
    llama_7b_helper,
    steering_vector_filenames,
    layers,
    multipliers,
    dataset,
    max_length=70,
):
    """
    llama_7b_helper: a llama_7b_helper object
    steering_vector_filenames: a dictionary mapping layer indices to steering vector filenames
    layers: a list of layer indices to test steering at
    multipliers: a list of multipliers to test multiplying steering vectors by
    dataset: a list of (question, best answer) pairs from TruthfulQA
    max_length: the maximum length of the generated text
    """
    for layer in layers:
        vec = torch.load(steering_vector_filenames[layer])
        vec = vec.to(llama_7b_helper.device)
        for multiplier in multipliers:
            pos_multiplier = multiplier
            neg_multiplier = -multiplier
            for q, a in tqdm(questions):
                llama_7b_helper.reset_all()
                llama_7b_helper.set_add_activations(layer, pos_multiplier * vec)
                sa = llama_7b_helper.generate_text(
                    "Question: " + q + " Answer: The answer is", max_length=max_length
                ).split("Answer: ")[1]
                llama_7b_helper.reset_all()
                llama_7b_helper.set_add_activations(layer, neg_multiplier * vec)
                na = llama_7b_helper.generate_text(
                    "Question: " + q + " Answer: The answer is", max_length=max_length
                ).split("Answer: ")[1]
                llama_7b_helper.reset_all()
                da = llama_7b_helper.generate_text(
                    "Question: " + q + " Answer: The answer is", max_length=max_length
                ).split("Answer: ")[1]
                results.append(
                    {
                        "answer_plus": sa,
                        "answer_minus": na,
                        "default_answer": da,
                        "question": q,
                        "correct_answer": a,
                    }
                )
            with open(
                f"./truthful_qa_results_layer_{layer}_multiplier_{pos_multiplier}_100q.json",
                "w",
            ) as rfile:
                json.dump(results, rfile)
