import torch
import random
import numpy as np


def knn(x_index, samples, k):
    samples = torch.Tensor(samples)
    distances = []
    for i in range(len(samples)):
        if i == x_index:
            distances.append(-1)
            continue

        distances.append((samples[x_index] - samples[i]).pow(2).sum().sqrt().item())

    _, top_indices = zip(*sorted(zip(distances, np.arange(len(distances)))))

    return top_indices[1:k+1]


def smote(samples, classes, augment_class):
    augment_index = random.choice([i for i in range(len(classes)) if classes[i] == augment_class])
    closest_index = knn(x_index=augment_index, samples=samples, k=1)[0]

    synthetic_sample = random.uniform(0, 1)*(samples[closest_index] + samples[augment_index])
    return synthetic_sample


samples = np.asarray([[1, 1], [3, 3], [2, 2]])
classes = [0, 1, 0]
augment_class = 0

synthetic_sample = smote(samples, classes, augment_class)
np.append(samples, synthetic_sample)
classes.append(augment_class)
