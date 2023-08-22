import numpy as np


def bce_loss(x, discrims, diffis, labels):
    loss = []
    for dis, dif, label in zip(discrims, diffis, labels):
        tmp = dis.item() * x
        logit = tmp - dif.item()
        prob = 1 / (1 + np.exp(-logit))
        label = label.item()
        loss.append(-(label * np.log(prob) + (1 - label) * np.log(1 - prob)))
    return np.array(loss).mean()


def bce_loss_multi(x, discrims, diffis, labels):
    loss = []
    for dis, dif, label in zip(discrims, diffis, labels):
        tmp = dis * x
        tmp = np.sum(tmp, axis=-1)
        logit = tmp - dif.squeeze(-1)
        prob = 1 / (1 + np.exp(-logit))
        label = label.item()
        loss.append(-(label * np.log(prob) + (1 - label) * np.log(1 - prob)))
    return np.array(loss).mean()
