import random


def randpol(positives, negatives, size):
    output = [0]*size
    for i in range(positives+negatives):
        if i < positives:
            output[i] = 1
        else:
            output[i] = -1
    random.shuffle(output)
    return output
