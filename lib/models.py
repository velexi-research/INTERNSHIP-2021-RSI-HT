import torch
from torch import nn
from torch.nn import functional as F
import datasets
import dataset_utils
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from abc import ABC
import os


def load_model_and_dataset(checkpoint_path):
    results = torch.load(checkpoint_path)
    conv1_weight = results['state_dict']['conv1.weight']

    if 'genes_dict' in results['args_dict'].keys():
        if conv1_weight.size(1) == 64:
            encoding = dataset_utils.encode_codon_seq
        else:
            encoding = dataset_utils.encode_base_seq

        k = results['args_dict']['k'] if 'k' in results['args_dict'].keys() else 500

        dataset = datasets.Genes('../data', k=k, genes_dict=results['args_dict']['genes_dict'], encoding=encoding)
    else:
        dataset = datasets.HistoneOccupancy(location='../data/h3_occupancy')

    model = TaxonomyCNN(dataset, kernel_size=conv1_weight.size(2)).eval()
    model.load_state_dict(results['state_dict'])

    return model, dataset


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

    plt.close()
    ax = plt.subplot()

    sns.heatmap(cm, annot=True, cmap="Blues", ax=ax)

    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)


def save_confusion_matrices_plots(results_location):
    for checkpoint_location in os.listdir(results_location):
        model, dataset = load_model_and_dataset(os.path.join(results_location, checkpoint_location))

        _, test_set = dataset_utils.split_dataset(dataset, test_size=0.1, shuffle=True)
        test_labels = torch.from_numpy(np.array(test_set.y))
        test_acc, y_pred = evaluate_model(model.cpu(), test_set, test_labels)

        cm = construct_confusion_matrix(y_true=test_labels.numpy(), y_pred=y_pred)
        plot_confusion_matrix(cm=cm, classes=dataset.genes_dict)
        plt.savefig('../reports/cm/' + checkpoint_location[:-3] + '.png')


def one_hot_encoding(output):
    output = output.transpose(1, 2)
    x = torch.zeros_like(output)

    one_hot_indices = torch.sum(output, 2)

    for batch_index in range(output.size(0)):
        for encoding_index in range(output.size(1)):
            one_hot_index = int(one_hot_indices[batch_index][encoding_index].item())

            if one_hot_index != 0:
                x[batch_index][encoding_index][one_hot_index-1] = 1

    x = x.transpose(1, 2)
    return x


def one_hot_decoding(encoding):
    return torch.argmax(encoding, dim=1)


def weight_averaging(model, previous_parameters):
    current_parameters = model.parameters()

    for p_current, p_previous in zip(current_parameters, previous_parameters):
        if p_current.requires_grad:
            with torch.no_grad():
                new_value = (p_current + p_previous)/2
                p_current.copy_(new_value)

    model.zero_grad()
    return model


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


class Autoencoder(nn.Module, ABC):
    def __init__(self, dataset, output_channels=2, kernel_size=2):
        super(Autoencoder, self).__init__()
        input_channels = dataset[0].size(0)
        seq_len = dataset[0].size(1)

        self.encoder_conv = nn.Conv1d(in_channels=input_channels, out_channels=output_channels,
                                      kernel_size=kernel_size, stride=1)
        self.encoder_dense = nn.Linear(seq_len//2-1, seq_len//4)

        self.encoder_pool = nn.MaxPool2d(2)

        self.decoder_conv = nn.ConvTranspose1d(in_channels=output_channels-1, out_channels=input_channels,
                                               kernel_size=kernel_size, stride=1)
        self.decoder_dense = nn.Linear(seq_len//4, seq_len-2)

    def forward(self, x):
        x = self.encoder_conv(x)
        x = F.leaky_relu(x)
        x = self.encoder_pool(x)

        x = self.encoder_dense(x)
        bottleneck = F.leaky_relu(x)

        x = self.decoder_dense(bottleneck)
        x = F.leaky_relu(x)

        x = self.decoder_conv(x)

        return x
