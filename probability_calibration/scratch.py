import json

with open("results.json", 'r') as dfile:
    data = json.load(dfile)
    question_set = set([d["question"] for d in data])
    dedpulicated_data = []
    for q in question_set:
        dedpulicated_data.append([d for d in data if d["question"] == q][0])
    print(len(dedpulicated_data), "data points")
    with open("results_deduplicated.json", 'w') as dfile:
        json.dump(dedpulicated_data, dfile)
    