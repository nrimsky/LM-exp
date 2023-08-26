import json
from random import shuffle

A_B = []
with open("refusal_data.json", "r") as f:
    data = json.load(f)
    for item in data:
        item["question"] = item["question"].strip()
        item["decline_answer"] = item["decline_answer"].strip()
        item["respond_answer"] = item["respond_answer"].strip()
        answers = [item["decline_answer"], item["respond_answer"]]
        shuffle(answers)
        if answers[0] == item["decline_answer"]:
            answer_matching_behavior = "(A)"
            answer_not_matching_behavior = "(B)"
        else:
            answer_matching_behavior = "(B)"
            answer_not_matching_behavior = "(A)"
        full_question = (
            f"{item['question']}\n\nChoices:\n(A) {answers[0]}\n(B) {answers[1]}"
        )
        A_B.append(
            {
                "question": full_question,
                "answer_matching_behavior": answer_matching_behavior,
                "answer_not_matching_behavior": answer_not_matching_behavior,
            }
        )

with open("refusal_data_A_B.json", "w") as f:
    json.dump(A_B, f)
