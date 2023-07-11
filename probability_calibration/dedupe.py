import json 

with open("questions.json") as qfile:
    data = json.load(qfile)
    question_set = set([d["question"] for d in data])
    dedpulicated_data = []
    for q in question_set:
        dedpulicated_data.append([d for d in data if d["question"] == q][0])
    print(len(dedpulicated_data), "data points")
    with open("questions_deduplicated.json", 'w') as qfile:
        json.dump(dedpulicated_data, qfile)