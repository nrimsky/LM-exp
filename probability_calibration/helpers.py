import math

def compute_probs(token_logprobs):
    probs = {'Y': 0, 'N': 0}
    total = 0
    for token, logprob in token_logprobs.items():
        prob = math.exp(logprob)
        if 'Y' in token.upper():
            probs['Y'] += prob
            total += prob 
        elif 'N' in token.upper():
            probs['N'] += prob
            total += prob
    for key in probs.keys():
        if total != 0:
            probs[key] /= total
        else:
            probs[key] = 0.0
    return probs

def ans_pos(tokens):
    for i, token in enumerate(tokens):
        if 'Y' in token.upper() or 'N' in token.upper():
            return i
    return -1