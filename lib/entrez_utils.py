from Bio import Entrez


def set_credentials():
    Entrez.name = 'yourncbimail@mail.com'
    Entrez.api_key = 'your entrez api key'
