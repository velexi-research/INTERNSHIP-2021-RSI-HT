from Bio import SeqIO
import argparse
import json
import numpy as np
import re
import random
import os

PARSER_ARGS = [
    {'name': '--db', 'type': str, 'default': 'snp'},
    {'name': '--query', 'type': str, 'default': 'brca2'},
    {'name': '--num_entries', 'type': int, 'default': 10000},
]

SIGNIFICANCE_LIST = [
    'benign',
    'benign-likely-benign',
    'likely-benign',
    'risk-factor',
    'conflicting-interpretations-of-pathogenicity',
    'likely-pathogenic',
    'pathogenic-likely-pathogenic',
    'pathogenic'
]


BASE_ENCODINGS = {
    'A': 0,
    'T': 1,
    'C': 2,
    'G': 3,
}


def get_args_dict():
    parser = argparse.ArgumentParser()
    for arg_dict in PARSER_ARGS:
        parser.add_argument(arg_dict['name'], type=arg_dict['type'], default=arg_dict['default'])
    args_dict = vars(parser.parse_args())
    return args_dict


def get_mutations_dict(file_name, ref_gene):
    with open(file_name, 'r') as file:
        records_dict = json.load(file)

    mutations_dict = {}
    for key in records_dict.keys():
        mutations = []
        start_index = len('HGVS=')

        for x in records_dict[key]['ExchangeSet']['DocumentSummary']['DOCSUM'].split(','):
            if ref_gene in x:
                if '[' not in x:
                    start_index = start_index + len(ref_gene) + 3
                    mutation = x[start_index:]

                    if 'N' not in mutation and 'R' not in mutation:
                        mutations.append(x[start_index:])
                    else:
                        mutations = []
                        break

                    start_index = 0
                else:
                    break

        if len(mutations) == 0:
            continue

        mutations_dict[key] = mutations
    return mutations_dict


def get_significance_dict(file_name):
    with open(file_name, 'r') as file:
        records_dict = json.load(file)

    significance_dict = {}
    for key in records_dict.keys():
        significance = records_dict[key]['ExchangeSet']['DocumentSummary']['CLINICAL_SIGNIFICANCE']
        significance_dict[key] = significance

    return significance_dict


def encode_significance_dict(significance_dict):
    encoded_dict = {}
    significance_indices = dict(zip(SIGNIFICANCE_LIST, np.arange(8)))
    for key in significance_dict.keys():
        current_encoding = np.zeros(8)
        for significance in significance_dict[key].split(','):
            if significance not in SIGNIFICANCE_LIST:
                current_encoding = None
                break
            current_encoding[significance_indices[significance]] = 1

        if current_encoding is not None:
            encoded_dict[key] = current_encoding

    return encoded_dict


def intersect_dicts(a, b):
    shared_keys = list(a.keys() & b.keys())

    new_a = {}
    new_b = {}

    for key in shared_keys:
        new_a[key] = a[key]
        new_b[key] = b[key]

    del a
    del b

    return new_a, new_b


def insert_substring(original_str, s, index):
    return original_str[:index] + s + original_str[index:]


def change_single_char(original_str, new_char, index):
    return original_str[:index] + new_char + original_str[index + 1:]


def preprocess_gene_data(file_name):
    with open(file_name, 'r') as file:
        ref_gene = SeqIO.parse(file, 'fasta')

        for record in ref_gene:
            ref_gene_info = re.split(':|-', record.id)

            ref_gene_name = ref_gene_info[0]
            ref_start_index = int(ref_gene_info[1])
            ref_end_index = int(ref_gene_info[2])

            ref_gene_seq = record.seq

    return {'name': ref_gene_name, 'start_index': ref_start_index, 'end_index': ref_end_index, 'seq': ref_gene_seq}


def delete_substring(original_str, start_index, end_index):
    return original_str[:start_index] + original_str[end_index + 1:]


def del_ins(original_str, ins, start_index, end_index):
    return original_str[:start_index] + ins + original_str[end_index + 1:]


def get_point_mutation_info(mutation):
    mutation_info = re.split('(A|T|C|G|R)', mutation)[:-1]
    mutation_index = int(mutation_info[0])
    original_base = mutation_info[1]
    new_base = mutation_info[3]

    return {'index': mutation_index, 'original_base': original_base, 'new_base': new_base}


def get_sequence_mutation_info(mutation):
    regex = '(delins|del|ins|dup)'
    mutation_info = [x for x in re.split(regex, mutation)]
    if '_' in mutation_info[0]:
        mutation_range = mutation_info[0].split('_')
        mutation_start_index = int(mutation_range[0])
        mutation_end_index = int(mutation_range[1])
    else:
        mutation_start_index, mutation_end_index = [int(mutation_info[0])] * 2

    operation_type = mutation_info[1]
    insert_subsequence = mutation_info[-1] if operation_type == 'ins' or operation_type == 'delins' else None

    return {'start_index': mutation_start_index, 'end_index': mutation_end_index,
            'operation_type': operation_type, 'insert_subsequence': insert_subsequence}


def apply_mutation(mutation, ref_gene_data):
    seq = str(ref_gene_data['seq'])
    seq_start_index = ref_gene_data['start_index']

    if not mutation.isupper():
        mutation_info = get_sequence_mutation_info(mutation)
        mutation_start_index = mutation_info['start_index'] - seq_start_index
        mutation_end_index = mutation_info['end_index'] - seq_start_index
        operation_type = mutation_info['operation_type']
        insert_subsequence = mutation_info['insert_subsequence']

        if operation_type == 'ins':
            seq = insert_substring(seq, insert_subsequence, mutation_end_index)
        elif operation_type == 'del':
            seq = delete_substring(seq, mutation_start_index, mutation_end_index)
        elif operation_type == 'dup':
            seq = insert_substring(seq, seq[mutation_start_index:mutation_end_index + 1], mutation_start_index)
        elif operation_type == 'delins':
            seq = del_ins(seq, insert_subsequence, mutation_start_index, mutation_end_index)
        else:
            print('Unknown operation type! ')
            return -1

        return seq, mutation_info
    else:
        mutation_info = get_point_mutation_info(mutation)
        mutation_index = mutation_info['index'] - seq_start_index
        original_base = mutation_info['original_base']
        new_base = mutation_info['new_base']

        if seq[mutation_index] == original_base:
            seq = change_single_char(seq, new_base, mutation_index)
            return seq, mutation_info
        else:
            print('Reference bases do not match!')
            return None, mutation_info


def get_random_mutated_k_mer(seq, mutation_info, k, ref_gene_start_index):
    x = random.randint(0, int(k / 2))
    if len(mutation_info.keys()) == 4:
        index = int((mutation_info['start_index']+mutation_info['end_index'])/2-ref_gene_start_index)
    else:
        index = mutation_info['index'] - ref_gene_start_index

    return seq[index-x:index+k-x]


def encode_seq(seq):
    encoding = np.zeros((len(seq), 4))

    for base, e in zip(seq, encoding):
        e[BASE_ENCODINGS[base]] = 1

    return encoding


def encode_significance_dict(significance_dict):
    encoded_significance_dict = {}
    for key in list(significance_dict.keys()):
        s = significance_dict[key]

        if 'benign' in s and 'pathogenic' in s or 'uncertain' in s:
            significance_dict.pop(key)
        else:
            if 'benign' in s:
                encoded_significance_dict[key] = 0
            elif 'pathogenic' in s:
                encoded_significance_dict[key] = 1
            else:
                significance_dict.pop(key)
    return encoded_significance_dict


def is_valid_seq(seq):
    for base in seq:
        if base not in list(BASE_ENCODINGS.keys()):
            return False
    return True


def serialize_json(d, location):
    with open(location, 'w') as file:
        json.dump(d, file, indent=4)


def train_test_split(x, y, test_size=0.1, shuffle=False):
    if len(x) != len(y):
        raise ValueError('Training data size mismatch between inputs and labels!')

    if shuffle:
        temp = list(zip(x, y))
        random.shuffle(temp)
        x, y = zip(*temp)

    num_samples = len(x)
    split_index = int(num_samples*(1-test_size))

    x_train, y_train = x[:split_index], y[:split_index]
    x_test, y_test = x[split_index:], y[split_index:]

    return (x_train, y_train), (x_test, y_test)


def main():
    gene_name = 'brca2'
    ref_code = 'NC_000013.11'
    k = 500

    gene_dir = os.path.join('../data', gene_name)
    records_location = os.path.join(gene_dir, 'records.json')
    gene_location = os.path.join(gene_dir, 'gene.fna')

    mutations_dict = get_mutations_dict(records_location, ref_code)
    significance_dict = get_significance_dict(records_location)

    mutations_dict, significance_dict = intersect_dicts(mutations_dict, significance_dict)

    gene_data = preprocess_gene_data(gene_location)

    mutations_keys = list(mutations_dict.keys())
    seq_save_location = os.path.join(gene_dir, 'mutations/np/')

    for key in mutations_keys:
        mutations_list = []

        for index, mutation in enumerate(mutations_dict[key]):
            seq, mutation_info = apply_mutation(mutation, gene_data)

            if seq is None:
                mutations_dict.pop(key)
                break

            seq_k_mer = get_random_mutated_k_mer(seq, mutation_info, k, gene_data['start_index'])
            seq_encoding = encode_seq(seq_k_mer)

            mutations_list.append(seq_encoding)

            current_save_location = os.path.join(seq_save_location, key + '.npy')
            if not os.path.exists(current_save_location):
                encoded_mutations_np = np.asarray(mutations_list)
                np.save(current_save_location, encoded_mutations_np)

    mutations_dict, significance_dict = intersect_dicts(mutations_dict, significance_dict)
    serialize_json(significance_dict, os.path.join(gene_dir, 'significance.json'))


if __name__ == '__main__':
    main()
