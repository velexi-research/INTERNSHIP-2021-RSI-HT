import torch
import shap
import numpy as np
import os
from deeplift.visualization import viz_sequence
from matplotlib import pyplot as plt
import dataset_utils
import datasets
import models
from PIL import Image


def decode_codon_seq_values(dinuc_shuff_explanation, original_k=500):
    triplet_indices = []
    indices_values = []
    nucleotides = []

    shap_values = np.transpose(dinuc_shuff_explanation)
    encodings_dict = dataset_utils.get_codon_encoding_dict()
    inv_encodings_dict = {v: k for k, v in encodings_dict.items()}
    for codon_triplet_index, codon_triplet in enumerate(shap_values):
        for codon_index in range(3):
            codon = codon_triplet[64*codon_index:64*(codon_index+1)]
            x = np.argmax(codon)
            if codon[x] == 0:
                x = np.argmin(codon)
                if codon[x] == 0:
                    x = -1
            if x != -1:
                triplet_indices.append(codon_triplet_index)
                indices_values.append(codon[x])
                nucleotides.append(str(inv_encodings_dict[x]))

    seq = ['K']*original_k
    v = np.zeros(original_k)

    for i in range(0, original_k-2, 3):
        v[i] += indices_values[i]
        v[i+1] += indices_values[i]
        v[i+1] += indices_values[i+1]
        v[i+2] += indices_values[i]
        v[i+2] += indices_values[i+1]
        v[i+2] += indices_values[i+2]
        seq[i] = nucleotides[i][0]
        seq[i+1] = nucleotides[i+1][0]
        seq[i+2] = nucleotides[i+2][0]

    seq[original_k-2] = nucleotides[original_k - 3][1]
    seq[original_k-1] = nucleotides[original_k - 3][2]

    v[original_k - 2] += indices_values[original_k - 4]
    v[original_k - 2] += indices_values[original_k - 3]
    v[original_k - 1] += indices_values[original_k - 3]

    seq = ''.join(seq)
    seq_encoded = dataset_utils.encode_base_seq(seq)
    for i in range(len(seq_encoded)):
        seq_encoded[i] = seq_encoded[i] * v[i]

    return seq_encoded


def general_plot(subplots_location):
    images = []

    for current_plot in os.listdir(subplots_location):
        img = Image.open(os.path.join(subplots_location, current_plot))
        images.append(np.asarray(img))

    fig, axes = plt.subplots(4, len(images)//4, figsize=(500, 500),
                             gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=True)

    for i in range(4):
        for j in range(len(images)//4):
            axes[i, j].axis('off')
            axes[i, j].imshow(images[i*4+j], aspect='auto')

    plt.subplots_adjust(0, 0, 1, 1)
    plt.show()


# histone_results_path = 'histone_29-07-2021_17:28:49.pt'
# genes_results_mode_3 = 'cnn_30-07-2021_00:55:51.pt'
# genes_results_codon_series = 'cnn_29-07-2021_18:07:13.pt'
# genes_results = 'cnn_28-07-2021_18:12:38.pt'

results_location = 'cnn_29-07-2021_18:07:13.pt'
results = torch.load(os.path.join('../results', results_location))
args_dict = results['args_dict']
plot_batch_size = 25

if args_dict['k'] == 500 and False:
    encoding = dataset_utils.encode_base_seq
else:
    encoding = dataset_utils.encode_codon_seq_series

if 'histone' in results_location:
    dataset = datasets.HistoneOccupancy('../data/h3_occupancy', encoding=dataset_utils.encode_codon_seq_series)
else:
    dataset = datasets.Genes('../data', k=args_dict['k'], genes_dict=results['args_dict']['genes_dict'], encoding=encoding)

dataset, test_set = dataset_utils.split_dataset(dataset, test_size=0.1, shuffle=True)
test_labels = torch.from_numpy(np.array(test_set.y))

model = models.TaxonomyCNN(dataset, kernel_size=results['state_dict']['conv1.weight'].size(2)).eval()
model.load_state_dict(results['state_dict'])

test_acc, y_pred = models.evaluate_model(model.cpu(), test_set, test_labels)

test_set_tensor = []
for entry in test_set:
    test_set_tensor.append(entry.numpy())

test_set_tensor = torch.from_numpy(np.asarray(test_set_tensor))

explainer = shap.GradientExplainer(model, test_set_tensor)
shap_values = explainer.shap_values(test_set_tensor[1].unsqueeze(0))[0]
shap_values = np.expand_dims(np.transpose(shap_values[0]).transpose(), 0)
test_tensor = np.expand_dims(np.transpose(test_set_tensor[1].numpy()).transpose(), 0)
shap_values = np.sum(shap_values, axis=-1)[:, :, None] * test_tensor

for values in shap_values:
    values_decoded = decode_codon_seq_values(values, original_k=args_dict['k'])
    for values_decoded_batch in np.split(values_decoded, args_dict['k']/plot_batch_size):
        viz_sequence.plot_weights(values_decoded_batch, subticks_frequency=1)
