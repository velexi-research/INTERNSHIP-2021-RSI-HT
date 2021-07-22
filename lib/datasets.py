import torch
import numpy as np
from Bio import SeqIO
from dataset_utils import split_dataset, encode_base_seq
import os


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
    def __init__(self, location, k, encoding=encode_base_seq):
        self.location = location
        self.gene_names = os.listdir(location)
        self. encoding = encoding

        self.x = []
        self.y = []

        gene_index = 0

        for gene in self.gene_names:
            if gene[0] == '.':
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
