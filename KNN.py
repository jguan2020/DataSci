import torch
import torch.nn.functional as F

def KNN(k, training_data, test_data, training_labels, test_labels):
    n = test_data.size(0)
    preds = torch.zeros(len(test_labels), dtype=torch.long)

    dist = torch.cdist(test_data, training_data)
    _, indices = torch.topk(dist, k, dim=1, largest=False)

    for t in range(n):
        labels = training_labels[indices[t]]
        most_common_label = torch.mode(labels).values.item()
        preds[t] = most_common_label

    return preds
