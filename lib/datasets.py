import torch
import numpy as np
from Bio import SeqIO
from dataset_utils import split_dataset, encode_base_seq
import os


DEFAULT_GENES_DICT = {
    'brca1': '672',
    'brca2': '675',
    'apc': '324',
    'pten': '5728',
    'vhl': '7428',
    'cdkn2a': '1029'
}


class Mutations(torch.utils.data.Dataset):
    def __init__(self, location):
        self.location = location
        self.all_mutations = os.listdir(location)

    def __len__(self):
        return len(self.all_mutations)

    def __getitem__(self, index):
        mutation_location = os.path.join(self.location, self.all_mutations[index])
        mutation = torch.from_numpy(np.load(mutation_location, allow_pickle=True).astype('float32'))

        return mutation


class Genes(torch.utils.data.Dataset):
    def __init__(self, location, k, encoding=encode_base_seq, genes_dict=None):
        self.location = location
        self.gene_names = os.listdir(location)
        self.encoding = encoding

        if genes_dict is None:
            genes_dict = DEFAULT_GENES_DICT
        self.genes_dict = genes_dict

        self.x = []
        self.y = []

        gene_index = 0

        for gene in self.gene_names:
            if gene not in self.genes_dict.keys():
                continue

            k_mers_location = os.path.join(location, gene + '/k_mers/' + str(k))
            num_k_mers = 0
            for k_mer in os.listdir(k_mers_location):
                self.x.append(os.path.join(k_mers_location, k_mer))
                num_k_mers += 1
            self.y.append([gene_index,]*num_k_mers)

            gene_index += 1

        self.y = np.array(self.y).flatten()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        k_mer_location = self.x[index]

        with open(k_mer_location, 'r') as file:
            fasta_records = SeqIO.parse(file, 'fasta')

            for record in fasta_records:
                seq = record.seq

        seq = self.encoding(seq, transpose=True)
        seq = torch.from_numpy(np.array(seq).astype('float32'))

        return seq


class HistoneOccupancy(torch.utils.data.Dataset):
    def __init__(self, location, encoding=encode_base_seq):
        self.location = location
        self.classes = os.listdir(location)
        self.encoding = encoding
        self.genes_dict = dict(zip(self.classes, self.classes))

        self.x = []
        self.y = []

        for c in self.classes:
            class_location = os.path.join(location, c)
            for seq_file in os.listdir(class_location):
                self.x.append(os.path.join(class_location, seq_file))
            self.y = self.y + ([int(c),]*len(os.listdir(class_location)))

        self.y = np.array(self.y).flatten()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        seq_location = self.x[index]

        with open(seq_location, 'r') as file:
            fasta_records = SeqIO.parse(file, 'fasta')

            for record in fasta_records:
                seq = record.seq

        seq = self.encoding(seq, transpose=True)
        seq = torch.from_numpy(np.array(seq).astype('float32'))

        return seq
