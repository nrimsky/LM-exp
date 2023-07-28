import requests
import json


def get_truthful_qa():
    qa = []
    for i in range(10):
        url = f"https://datasets-server.huggingface.co/rows?dataset=truthful_qa&config=generation&split=validation&offset={i * 100}&limit=100"
        r = requests.get(url).text
        t_qa_data = [row["row"] for row in json.loads(r)["rows"]]
        for d in t_qa_data:
            qa.append((d["question"], d["best_answer"]))
    return qa
