"""
Run EpHod to predict pHopt for enzyme sequences
"""


import numpy as np
import pandas as pd
from sklearn.svm import SVR
import torch
from torch.nn.parallel import DataParallel
import torch.nn as nn
import random
import tqdm
import argparse
import joblib
import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

import esm

sys.path.insert(1, './')
from ephod.training import nn_models 




def parse_arguments():
    '''Parse command-line training arguments'''
    
    parser = argparse.ArgumentParser(description="Predict pHopt of enzymes with EpHod")
    parser.add_argument('--fasta_path', type=str,  
                        help='Path to fasta file of enzyme sequences')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='Directory to which prediction results will be written')
    parser.add_argument('--csv_name', type=str, default='prediction.csv', 
                        help='Name of csv file to which prediction results will be written')
    parser.add_argument('--verbose', default=1, type=int,
                        help='Whether to print out prediction progress to terminal')
    parser.add_argument('--save_attention_weights', default=0, type=int,
                        help="Whether to write RLAT attention weights for each sequence")
    parser.add_argument('--save_embeddings', default=0, type=int,
                        help="Whether to save 2560-dim EpHod embeddings for each sequence")
    args = parser.parse_args()

    return args




def write_attention_weights(accs, seqs, attention_weights, attention_dir, attention_mode='average'):
    '''Write RLAT attention weights for each sequence'''
    
    for i, (acc,seq) in enumerate(zip(accs, seqs)):
        seqlen = len(seq)
        weights = attention_weights[i,:,:seqlen]
        if attention_mode == 'average':
            weights = weights.mean(axis=0).transpose()
        elif attention_mode == 'max':
            weights = weights.max(axis=0).transpose()
        else:
            raise ValueError("attention_mode must be either 'average' or 'max'")
        weights = pd.DataFrame(weights.transpose(), index=list(seq), columns=['weights'])
        weights.to_csv(f'{attention_dir}/{acc}.csv')
    
        

               
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




def replace_noncanonical(seq, replace_char='X'):
    '''Replace all non-canonical amino acids with a specific character'''

    for char in ['B', 'J', 'O', 'U', 'Z']:
        seq = seq.replace(char, replace_char)
    return seq




class EpHodModel():
    
    def __init__(self, seed=0):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device != 'cuda':
            print('WARNING: You are not using a GPU. Inference will be slow')
        self.set_seed(seed=seed)
        self.esm1v_model, self.esm1v_batch_converter = self.load_ESM1v_model()
        self.svr_model, self.svr_stats = self.load_SVR_model()
        self.rlat_model = self.load_RLAT_model()
        self.esm1v_model.eval()
        self.rlat_model.eval()

    
    def set_seed(self, seed):
    
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
        
    def load_ESM1v_model(self):
        '''Return pretrained ESM1v model weights and batch converter'''
        
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
        model = model.to(self.device)
        batch_converter = alphabet.get_batch_converter()
        
        return model, batch_converter
    
    
    def get_ESM1v_embeddings(self, accs, seqs):
        '''Return per-residue embeddings (padded) for protein sequences from ESM1v model'''

        seqs = [replace_noncanonical(seq, 'X') for seq in seqs]
        data = [(accs[i], seqs[i]) for i in range(len(accs))]
        batch_labels, batch_strs, batch_tokens = self.esm1v_batch_converter(data)
        batch_tokens = batch_tokens.to(device=self.device, non_blocking=True)
        emb = self.esm1v_model(batch_tokens, repr_layers=[33], return_contacts=False)
        emb = emb["representations"][33]
        emb = emb.transpose(2,1) # From (batch, seqlen, features) to (batch, features, seqlen)

        return emb
    
    
    def load_RLAT_model(self):
        '''Return residual light attention top model'''
        
        model = nn_models.ResidualLightAttention(dim=1280, kernel_size=7, dropout=0.0, res_blocks=4, activation='elu')
        model = model.to(self.device)
        url = 'https://zenodo.org/records/14252615/files/ESM1v-RLATtr.pt?download=1'
        model_dict = torch.hub.load_state_dict_from_url(url, progress=False, map_location=self.device)
        model_dict = {key[len('module.'):]: value for key, value in model_dict.items()} # Remove DataParallel suffix
        model.load_state_dict(model_dict)

        return model

    
    def load_SVR_model(self):
        '''Return SVR top model'''
        
        this_dir, this_filename = os.path.split(__file__)
        path = os.path.join(this_dir, 'data', 'ESM1v-SVR.pkl')
        svr_model, svr_stats = joblib.load(path)

        return svr_model, svr_stats
        
    
    def predict(self, accs, seqs):
        '''Predict pHopt of sequences with EpHod'''
        
        # Get ESM1v embeddings and run RLATtr model
        emb_esm1v = self.get_ESM1v_embeddings(accs, seqs)
        maxlen = emb_esm1v.shape[-1]
        masks = [[1] * len(seqs[i]) + [0] * (maxlen - len(seqs[i])) \
                 for i in range(len(seqs))]
        masks = torch.tensor(masks, dtype=torch.int32)
        masks = masks.to(self.device)
        out = self.rlat_model(emb_esm1v, masks)
        rlat_pred, rlat_emb, rlat_attn = [item.cpu().numpy() for item in out]
    
        # Run SVR
        emb_pool = emb_esm1v.cpu().numpy().mean(axis=-1) # (batch, features, seqlen)
        emb_pool = (emb_pool - self.svr_stats[:,0]) / (self.svr_stats[:,1] + 1e-8) # Normalize with means/std.dev
        svr_pred = self.svr_model.predict(emb_pool) # Note that batch size > 1 affects this pooling
        ensemble_pred = (rlat_pred + svr_pred) / 2
        outdict = dict(rlat_pred=rlat_pred, rlat_emb=rlat_emb, rlat_attn=rlat_attn, 
                       svr_pred=svr_pred, ensemble_pred=ensemble_pred)

        return outdict
            


    
def main():
    '''Run inference with EpHod model'''
    
    args = parse_arguments()
    
    # Read enzyme sequence data
    assert os.path.exists(args.fasta_path), f"File not found in {args.fasta_path}"
    headers, sequences = read_fasta(args.fasta_path)
    accessions = [head.split()[0] for head in headers]
    headers, sequences, accessions = [np.array(item) for item in (headers, sequences, accessions)]
    assert len(accessions) == len(headers) == len(sequences), 'Fasta file has unequal headers and sequences'
    numseqs = len(sequences)
    if args.verbose:
        print(f'Reading {numseqs} sequences from {args.fasta_path}')
        
    # Check sequence lengths
    lengths = np.array([len(seq) for seq in sequences])
    if max(lengths) > 1022:
        long_count = np.sum(lengths > 1022)
        warning = f"{long_count} sequences are longer than 1022 residues and will be truncated"
        sequences = np.asarray([item[:1022] for item in sequences])    
    
    # Directory to write pHopt predictions
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    phout_file = f'{args.save_dir}/{args.csv_name}'  # Prediction output file
    
    # Directory to write RLATtr attention weights
    if args.save_attention_weights:
        attention_dir = f'{args.save_dir}/attention_weights'
        if not os.path.exists(attention_dir):
            os.makedirs(attention_dir)

    # CSV file to write EpHod embeddings
    embed_file = f'{args.save_dir}/embeddings.csv'

    # Initialize EpHod model
    ephod_model = EpHodModel()
    if args.verbose:
        print('Initializing EpHod model')
        print(f'Device is {ephod_model.device}')

    # Batch prediction
    batch_size = 1 # Use batch_size of 1 
    num_batches = int(np.ceil(numseqs / batch_size))
    all_ypred, all_emb_ephod = np.empty((0,3)), np.empty((0, 2560))
    
    with torch.no_grad():
        batches = range(num_batches)
        if args.verbose:
            batches = tqdm.tqdm(batches, desc="Predicting pHopt")
        
        for batch_step in batches:
            
            # Batch sequences
            start_idx = batch_step * batch_size
            stop_idx = (batch_step + 1) * batch_size
            accs = accessions[start_idx : stop_idx] 
            seqs = sequences[start_idx : stop_idx]
            
            # Predict with EpHod model
            out = ephod_model.predict(accs, seqs) # dict_keys(['rlat_pred', 'rlat_emb', 'rlat_attn', 'svr_pred', 'ensemble_pred'])
            all_ypred = np.vstack((all_ypred, np.array([out['rlat_pred'], out['svr_pred'], out['ensemble_pred']]).transpose()))
            all_emb_ephod = np.vstack((all_emb_ephod, out['rlat_emb']))
            if args.save_attention_weights:
                _ = write_attention_weights(accs, seqs, out['rlat_attn'], attention_dir)
            
    if args.save_embeddings:
        all_emb_ephod = pd.DataFrame(np.array(all_emb_ephod), index=accessions)
        all_emb_ephod.to_csv(embed_file)

    if args.verbose:
        print('Prediction completed.')
        
    # Save predictions
    all_ypred = pd.DataFrame(all_ypred, index=accessions, columns=['RLATtr', 'SVR', 'Ensemble'])
    all_ypred.to_csv(phout_file)
    
    
    

if __name__ == '__main__':
    main()
    
        
        
        
        



    
