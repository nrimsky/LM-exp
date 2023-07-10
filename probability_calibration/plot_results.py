import json 
from matplotlib import pyplot as plt

def get_results():
    with open("results.json") as dfile:
        dataset = json.load(dfile)
        results = []
        results_words = []
        for item in dataset:
            logit_probs_just_y = item["logit_probs_just_yn"]["Y"]
            logit_probs_full_q = item["logit_probs_full_q"]["Y"]
            prob_said = item["probability_said_full_q"]
            logit_probs_words = item["logit_probs_full_q_words"]["Y"]
            probs_words = item["probability_said_full_q_words"]
            results.append((logit_probs_just_y, logit_probs_full_q, prob_said/100))
            results_words.append((logit_probs_just_y, logit_probs_words, probs_words/100))
        print(len(results), "data points")
        return results, results_words
    
def plot_results(results):
    """
    Create 3 scatter subplots, one for each of the following:
    1. Probability said vs. logit probability of Y from just Y/N
    2. Probability said vs. logit probability of Y from full question
    3. Logit probability of Y from just Y/N vs. logit probability of Y from full question
    Save the figure as "results.png"
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Probability said vs. logit probability of Y from just Y/N
    axs[0].scatter([x[0] for x in results], [x[2] for x in results])
    axs[0].set_xlabel("Logit probability of Y from just Y/N")
    axs[0].set_ylabel("Probability said")
    # Probability said vs. logit probability of Y from full question
    axs[1].scatter([x[1] for x in results], [x[2] for x in results])
    axs[1].set_xlabel("Logit probability of Y from full question")
    axs[1].set_ylabel("Probability said")
    # Logit probability of Y from just Y/N vs. logit probability of Y from full question
    axs[2].scatter([x[0] for x in results], [x[1] for x in results])
    axs[2].set_xlabel("Logit probability of Y from just Y/N")
    axs[2].set_ylabel("Logit probability of Y from full question")
    plt.savefig("results.png")

def plot_results_words(results):
    """
    Create 3 scatter subplots, one for each of the following:
    1. Word-based probability vs. logit probability of Y from just Y/N
    2. Word-based probability vs. logit probability of Y from full question
    3. Logit probability of Y from just Y/N vs. logit probability of Y from full question
    Save the figure as "results.png"
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Probability said vs. logit probability of Y from just Y/N
    axs[0].scatter([x[0] for x in results], [x[2] for x in results])
    axs[0].set_xlabel("Logit probability of Y from just Y/N")
    axs[0].set_ylabel("Probability expressed with words")
    # Probability said vs. logit probability of Y from full question
    axs[1].scatter([x[1] for x in results], [x[2] for x in results])
    axs[1].set_xlabel("Logit probability of Y from full question")
    axs[1].set_ylabel("Probability expressed with words")
    # Logit probability of Y from just Y/N vs. logit probability of Y from full question
    axs[2].scatter([x[0] for x in results], [x[1] for x in results])
    axs[2].set_xlabel("Logit probability of Y from just Y/N")
    axs[2].set_ylabel("Logit probability of Y from full question")
    plt.savefig("results_word_probabilities.png")

if __name__ == "__main__":
    results, results_words = get_results()
    plot_results(results)
    plot_results_words(results_words)



    