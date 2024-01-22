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


def NB(a, training_data, test_data, training_labels, test_labels):
    i1 = torch.where(training_labels == 1)[0]
    i2 = torch.where(training_labels == 2)[0]
    training_data1 = training_data[i1, :]
    training_data2 = training_data[i2, :]

    n = len(training_data)
    n1, x = i1.size(0), training_data.size(1)
    n2, x = i2.size(0), training_data.size(1)
    S1 = torch.matmul((training_data1 - training_data1.mean(dim=0, keepdim=True)).t(), (training_data1 - training_data1.mean(dim=0, keepdim=True))) / (training_data1.size(0) - 1)
    S2 = torch.matmul((training_data2 - training_data2.mean(dim=0, keepdim=True)).t(), (training_data2 - training_data2.mean(dim=0, keepdim=True))) / (training_data2.size(0) - 1)
    m1 = torch.mean(training_data1, dim=0)
    m2 = torch.mean(training_data2, dim=0)

    total = n1 + n2
    preds = torch.zeros(len(test_labels))

    pc1 = n1 / total
    pc2 = n2 / total

    for i in range(len(test_data)):
        g1x = -total / 2 * torch.log(2 * torch.tensor([3.1416])) - torch.log(torch.det(S1)) / 2 - \
              1 / 2 * (test_data[i, :] - m1) @ torch.inverse(S1) @ torch.transpose(test_data[i, :] - m1, 0, 0) + torch.log(torch.tensor(pc1))

        g2x = -total / 2 * torch.log(2 * torch.tensor([3.1416])) - torch.log(torch.det(S2)) / 2 - \
              1 / 2 * (test_data[i, :] - m2) @ torch.inverse(S2) @ torch.transpose(test_data[i, :] - m2, 0, 0) + torch.log(torch.tensor(pc2))

        if g1x > g2x:
            preds[i] = 1
        else:
            preds[i] = 2

    return preds
