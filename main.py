import torch
import pandas as pd
from functions import KNN,NB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from Excel
dataset = pd.read_excel("dataset.xlsx")
labels = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:].values

# Set parameters
n = 150
k = 9
a = 100
trials = 100
totalNB = 0
totalKNN = 0
totalSim = 0

for t in range(trials):
    # Generate random indices for training and testing sets
    randn = torch.randperm(191)[:n].tolist()
    itest = []

    for i in range(191 - n):
        for j in range(191):
            if j not in randn and j not in itest:
                itest.append(j)
                break

    test_data = torch.tensor(data[itest, :])
    training_data = torch.tensor(data[randn, :])
    test_labels = torch.tensor(labels[itest])
    training_labels = torch.tensor(labels[randn])

    # Convert labels
    i1 = torch.where(training_labels >= a)[0]
    i2 = torch.where(training_labels < a)[0]
    training_labels[i1] = 1
    training_labels[i2] = 2

    for i in range(191 - n):
        if test_labels[i] >= a:
            test_labels[i] = 1
        else:
            test_labels[i] = 2

    # Run KNN and NB
    knn_preds = KNN(k, training_data, test_data, training_labels, test_labels)
    nb_preds = NB(a, training_data, test_data, training_labels, test_labels)

    # Calculate similarity and accuracy
    similarity = torch.sum(nb_preds == knn_preds) / (191 - n)
    nb_acc = accuracy_score(test_labels.numpy(), nb_preds.numpy())
    knn_acc = accuracy_score(test_labels.numpy(), knn_preds.numpy())

    totalSim = totalSim + similarity.item()
    totalNB = totalNB + nb_acc
    totalKNN = totalKNN + knn_acc
    

avgSim = totalSim / trials
avgNB = totalNB / trials
avgKNN = totalKNN / trials

print("Over 100 trials, the results are")
print(f"Similarity between KNN and NB predictions: {avgSim}")
print(f"Naive Bayes Accuracy: {avgNB}")
print(f"KNN Accuracy: {avgKNN}")

