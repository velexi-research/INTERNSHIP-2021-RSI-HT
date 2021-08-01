import torch
import numpy as np
from Bio import SeqIO
import dataset_utils
import os
import random


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
    def __init__(self, location, args_dict, transpose_output=True):
        self.location = location
        self.gene_names = os.listdir(location)
        self.transpose_output = transpose_output

        self.encoding = dataset_utils.encode_base_seq
        self.genes_dict = args_dict['genes_dict']

        if 'encoding' in args_dict.keys():
            if args_dict['encoding'] == 'sequential':
                self.encoding = dataset_utils.encode_codon_seq_series

        self.shuffle = False

        self.x = []
        self.y = []

        gene_index = 0

        for gene in self.gene_names:
            if gene not in self.genes_dict.keys():
                continue

            k_mers_location = os.path.join(location, gene + '/k_mers/' + str(args_dict['k']))
            num_k_mers = 0
            for k_mer in os.listdir(k_mers_location):
                self.x.append(os.path.join(k_mers_location, k_mer))
                num_k_mers += 1
            self.y.append([gene_index]*num_k_mers)

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

        if self.shuffle:
            seq_list = list(seq)
            random.shuffle(seq_list)
            seq = ''.join(seq_list)

        seq = self.encoding(seq)
        if self.transpose_output:
            seq = np.transpose(seq)

        seq = torch.from_numpy(np.array(seq).astype('float32'))

        return seq


class HistoneOccupancy(torch.utils.data.Dataset):
    def __init__(self, location, encoding=dataset_utils.encode_base_seq):
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

        seq = self.encoding(seq)
        seq = np.transpose(seq)

        seq = torch.from_numpy(seq.astype('float32'))

        return seq
