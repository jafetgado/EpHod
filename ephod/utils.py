"""
Utility functions for processing and analysis
Author: Japheth Gado
"""



import numpy as np
import json



               
def read_fasta(fasta, return_as_dict=False):
    '''Read the protein sequences in a fasta file. If return_as_dict, return a dictionary
    with headers as keys and sequences as values, else return a tuple, 
    (list_of_headers, list_of_sequences)'''
    
    headers, sequences = [], []

    with open(fasta, 'r') as fast:
        
        for line in fast:
            if line.startswith('>'):
                head = line.replace('>','').strip()
                headers.append(head)
                sequences.append('')
            else :
                seq = line.strip()
                if len(seq) > 0:
                    sequences[-1] += seq

    if return_as_dict:
        return dict(zip(headers, sequences))
    else:
        return (headers, sequences)






def write_fasta(headers, seqdata, path):
    '''Write a fasta file (path) from a list of headers and a corresponding list 
    of sequences (seqdata)'''
    
    with open(path, 'w') as pp:
        for i in range(len(headers)):
            pp.write('>' + headers[i] + '\n' + seqdata[i] + '\n')
    
    return





def write_json(writedict, path, indent=4, sort_keys=False):
    '''Save dictionary as json file in path'''
    
    f = open(path, 'w')
    _ = f.write(json.dumps(writedict, indent=indent, sort_keys=sort_keys))
    f.close()
    
    




def read_json(path):
    '''Return a dictionry read from a json file'''
    
    f = open(path, 'r')
    readdict = json.load(f)
    f.close()
    
    return readdict 






def replace_noncanonical(seq, replace_char='X'):
    '''Replace all non-canonical amino acids with a specific character'''

    for char in ['B', 'J', 'O', 'U', 'Z']:
        seq = seq.replace(char, replace_char)
    return seq






def get_amino_composition(seq, normalize=True):
    '''Return the amino acid composition for a protein sequence'''

    aac = np.array([seq.count(amino) for amino in list('ACDEFGHIKLMNPQRSTVWY')])
    if normalize:
        aac = aac / len(seq)

    return aac


