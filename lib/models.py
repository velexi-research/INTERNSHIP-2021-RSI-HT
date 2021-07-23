import torch
from torch import nn
from torch.nn import functional as F
import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from abc import ABC


def evaluate_model(model, x, y):
    model = model.eval()
    predictions_list = []

    if type(type(x) == datasets.Genes):
        correct_predictions = 0
        for x_entry, y_entry in zip(x, y):
            prediction = model(x_entry.unsqueeze(0))
            y_pred = torch.argmax(prediction)

            if y_pred == y_entry:
                correct_predictions += 1

            predictions_list.append(y_pred.item())

        return 1.0*correct_predictions / y.size(0), np.asarray(predictions_list)

    if type(x) != torch.Tensor:
        x = torch.from_numpy(np.asarray(x).astype('float32')).cuda()

    if type(y) != torch.Tensor:
        y = torch.from_numpy(np.asarray(y).astype('float32')).cuda()

    with torch.no_grad():
        prediction = model(x)
        y_pred = torch.round(prediction).squeeze()
        accuracy = 1.0*torch.sum(torch.eq(y_pred, y)) / y.size(0)

    return accuracy.cpu(), y_pred


def construct_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize='pred')
    return cm


def plot_confusion_matrix(cm, classes=datasets.DEFAULT_GENES_DICT):
    if type(classes) == dict:
        classes = list(classes)

    ax = plt.subplot()

    sns.heatmap(cm, annot=True, cmap="Blues", ax=ax)

    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    # plt.show()


class TaxonomyCNN(nn.Module, ABC):
    def __init__(self, dataset, output_channels=2, kernel_size=2):
        super(TaxonomyCNN, self).__init__()
        input_channels = dataset[0].size(0)
        seq_len = dataset[0].size(1)

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels,
                               kernel_size=kernel_size, stride=1)

        self.pool1 = nn.MaxPool2d(2)

        self.dense1 = nn.Linear(seq_len//2-1, seq_len//4)
        self.dense2 = nn.Linear(seq_len//4, seq_len//8)
        self.dense3 = nn.Linear(seq_len//8, len(dataset.genes_dict.keys()))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = torch.flatten(x, 1)

        x = self.dense1(x)
        x = F.relu(x)

        x = self.dense2(x)
        x = F.relu(x)

        x = self.dense3(x)

        output = F.softmax(x, dim=1)
        return output
