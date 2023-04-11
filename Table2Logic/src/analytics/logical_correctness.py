import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



def mean_correct():
    with open('experiments/exp__20220328_135352/logically_correct_per_batch.json') as json_file:
        data = json.load(json_file)

    simplified = []
    for epoch in data:
        simplified.append({"train": np.mean(epoch["train"]), "val": np.mean(epoch["val"])})

    train = [x["train"] for x in simplified]
    val = [x["val"] for x in simplified]

    d = {'correct': train + val, 'epoch': [x for x in range(len(train))] + [x for x in range(len(val))],
         'dataset': ['train'] * len(train) + ['val'] * len(val)}
    df = pd.DataFrame(data=d)

    sns.relplot(x="epoch", y="correct", hue="dataset", kind="line", data=df)
    plt.show()


def at_least_one():
    with open('experiments/exp__20220328_135352/logically_correct_per_batch.json') as json_file:
        data = json.load(json_file)

    simplified = []
    for epoch in data:
        simplified.append({"train": 1-(np.sum([1 if x==0 else 0 for x in epoch["train"]])/len(epoch["train"])), "val": 1-(np.sum([1 if x==0 else 0 for x in epoch["val"]])/len(epoch["val"]))})

    train = [x["train"] for x in simplified]
    val = [x["val"] for x in simplified]

    d = {'correct': train + val, 'epoch': [x for x in range(len(train))] + [x for x in range(len(val))],
         'dataset': ['train'] * len(train) + ['val'] * len(val)}
    df = pd.DataFrame(data=d)

    sns.relplot(x="epoch", y="correct", hue="dataset", kind="line", data=df)
    plt.ylim(0, 1.0)
    plt.show()

if __name__ == '__main__':
    at_least_one()