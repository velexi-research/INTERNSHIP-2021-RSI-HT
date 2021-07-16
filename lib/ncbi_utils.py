from Bio import Entrez
import xmltodict
import requests
import zipfile
import io
import os
import shutil

DEFAULT_SEARCH_PARAMETERS = {
    'usehistory': ' Y',
    'rettype': 'xml',
    'retmode': 'text',
    'retmax': 100
}


GENE_DOWNLOAD_QUERY = 'https://api.ncbi.nlm.nih.gov/datasets/v1alpha/gene/id/' \
                      '{gene_id}/download?include_annotation_type=FASTA_GENE&filename=temp.zip '


def search(db, query, search_parameters=None):
    if search_parameters is None:
        search_parameters = DEFAULT_SEARCH_PARAMETERS

    search_handle = Entrez.esearch(db=db, term=query, **search_parameters)
    search_record = Entrez.read(search_handle)
    search_handle.close()

    final_record = {}
    for entry_id in search_record['IdList']:
        fetch_handle = Entrez.efetch(db=db, id=entry_id, retmode='xml')
        final_record[entry_id] = xmltodict.parse(fetch_handle.read())
        fetch_handle.close()

    return final_record


def download_nucleotide(file_name, nucleotide_id, return_type='fasta', return_mode='text'):
    handle = Entrez.efetch(
        db='nucleotide', id=nucleotide_id, rettype=return_type, retmode=return_mode
    )

    with open(file_name, 'w') as file:
        file.write(handle.read())

    handle.close()
    print('Nucleotide ' + nucleotide_id + ' data successfully serialized as ' + file_name)


def download_gene(file_name, gene_id):
    temp_dir = '../data/temp'
    r = requests.get(GENE_DOWNLOAD_QUERY.format(gene_id=gene_id), stream=True)

    zip_file = zipfile.ZipFile(io.BytesIO(r.content))
    zip_file.extractall(temp_dir)

    shutil.move(os.path.join(temp_dir, 'ncbi_dataset/data/gene.fna'), file_name)
    shutil.rmtree(temp_dir)
    print('Gene ' + gene_id + ' data successfully serialized as ' + file_name)
