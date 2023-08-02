from questions import questions
from pquote import get_letter_probs_and_quoted_probs_for_question as pquote
from wquote import get_letter_probs_and_quoted_probs_for_question as wquote
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    presults = []
    wresults = []
    for question in tqdm(questions):
        presults.append(pquote(question))
        wresults.append(wquote(question))
    plot(presults, "pquote.png")
    plot(wresults, "wquote.png")

def plot(results, fname):
    logit_probs = [max(result[0].values())*100 for result in results]
    quoted_probs = [result[2] for result in results]
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.scatter(logit_probs, quoted_probs, alpha=0.5)
    plt.xlabel("Internal Probability")
    plt.ylabel("Quoted Probability")
    plt.savefig(fname)

if __name__ == "__main__":
    main()