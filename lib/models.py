import torch
from torch import nn
from torch.nn import functional as F
import datasets
import numpy as np
from abc import ABC


def evaluate_model(model, x, y):
    model = model.eval()

    if type(type(x) == datasets.Genes):
        correct_predictions = 0
        for x_entry, y_entry in zip(x, y):
            prediction = model(x_entry.unsqueeze(0))
            y_predicted = torch.argmax(prediction)

            if y_predicted == y_entry:
                correct_predictions += 1

        return 1.0*correct_predictions / y.size(0)

    if type(x) != torch.Tensor:
        x = torch.from_numpy(np.asarray(x).astype('float32')).cuda()

    if type(y) != torch.Tensor:
        y = torch.from_numpy(np.asarray(y).astype('float32')).cuda()

    with torch.no_grad():
        prediction = model(x)
        predicted_labels = torch.round(prediction).squeeze()
        accuracy = 1.0*torch.sum(torch.eq(predicted_labels, y)) / y.size(0)

    return accuracy.cpu()


class TaxonomyCNN(nn.Module, ABC):
    def __init__(self):
        super(TaxonomyCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=2, stride=1)

        self.pool1 = nn.MaxPool2d(2)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.dense1 = nn.Linear(249, 83)
        self.dense2 = nn.Linear(83, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.dense2(x)

        output = F.softmax(x)
        return output
