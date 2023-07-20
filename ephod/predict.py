"""Predict pHopt for enzyme sequences with EpHod from command line
Author: Japheth Gado
"""



import numpy as np
import pandas as pd
from sklearn.svm import SVR

import torch

import tqdm
import time
import argparse
import joblib
import os
import subprocess
import sys

sys.path.insert(1, './')
import ephod.utils as utils
import ephod.models as models
print = models.print # Flush printing

import warnings
warnings.filterwarnings('ignore')








def parse_arguments():
    '''Parse command-line training arguments'''
    
    parser = argparse.ArgumentParser(description="Predict pHopt of enzymes with EpHod")
    parser.add_argument('--fasta_path', type=str,  
                        help='Path to fasta file of enzyme sequences')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='Directory to which prediction results will be written')
    parser.add_argument('--csv_name', type=str, default='prediction.csv', 
                        help='Name of csv file to which prediction results will be written')
    parser.add_argument('--aac_svr', type=int, default=0, 
                        help='If 1, use the simple AAC-SVR model instead of EpHod')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size used in inference')
    parser.add_argument('--verbose', default=1, type=int,
                        help='Whether to print out prediction progress to terminal')
    parser.add_argument('--save_attention_weights', default=0, type=int,
                        help="Whether to write light attention weights for each sequence")
    parser.add_argument('--attention_mode', default='average', type=str,
                        help="Either 'average' or 'max'. How to derive Lx1 weights from Lx1280 tensor")
    parser.add_argument('--save_embeddings', default=0, type=int,
                        help="Whether to save 2560-dim EpHod embeddings for each sequence")
    args = parser.parse_args()

    return args








def write_attention_weights(args, accs, seqs, attention_weights, attention_dir):
    '''Write attention weights for each sequence'''
    
    for i, (acc,seq) in enumerate(zip(accs, seqs)):
        seqlen = len(seq)
        weights = attention_weights[i,:,:seqlen].to('cpu').detach().numpy()
        if args.attention_mode == 'average':
            weights = weights.mean(axis=0).transpose()
        elif args.attention_mode == 'max':
            weights = weights.max(axis=0).transpose()
        else:
            raise ValueError("attention_mode must be either 'average' or 'max'")
        weights = pd.DataFrame(weights.transpose(), index=list(seq), columns=['weights'])
        weights.to_csv(f'{attention_dir}/{acc}.csv')
    
        






def predict_aac_svr(args, accessions, sequences):
    '''Predict pHopt with a support vector regression model using amino acid composition'''
    
    # Load models
    this_dir, this_filename = os.path.split(__file__)
    model_path = os.path.join(this_dir, 'saved_models', 'AAC-SVR', 'aac_svr.pkl')
    stats_path = os.path.join(this_dir, 'saved_models', 'AAC-SVR', 'mean_std.csv')
    if not os.path.exists(model_path):
        _ = models.download_models()
    aac_svr_model = joblib.load(model_path)
    stats = pd.read_csv(stats_path, index_col=0)
    means, stds = stats['means'].values, stats['stds'].values
    
    # Predict pHopt from amino acid composition of sequences with SVR
    aac = np.array([utils.get_amino_composition(seq) for seq in sequences])
    aac = (aac - means) / (stds + 1e-8)
    ypred = aac_svr_model.predict(aac)

    return ypred


 
    
    


    
def main():
    '''Run inference with EpHod model'''
    
    args = parse_arguments()

    
    # Read enzyme sequence data
    assert os.path.exists(args.fasta_path), f"File not found in {args.fasta_path}"
    headers, sequences = utils.read_fasta(args.fasta_path)
    accessions = [head.split()[0] for head in headers]
    headers, sequences, accessions = [np.array(item) for item in \
                                      (headers, sequences, accessions)]
    assert len(accessions) == len(headers) == len(sequences), 'Fasta file has unequal headers and sequences'
    numseqs = len(sequences)
    if args.verbose:
        print(f'Reading {numseqs} sequences from {args.fasta_path}')
        
    
    # Check sequence lengths
    lengths = np.array([len(seq) for seq in sequences])
    long_count = np.sum(lengths > 1022)
    warning = f"{long_count} sequences are longer than 1022 residues and will be omitted"
    
    # Omit sequences longer than 1022
    if max(lengths) > 1022:
        print(warning)
        locs = np.argwhere(lengths <= 1022).flatten()
        headers, sequences, accessions = [array[locs] for array in \
                                          (headers, sequences, accessions)]
        numseqs = len(sequences)
    
    
    # Prepare files/directories for writing predictions
    # First, prepare output directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Prediction output file
    phout_file = f'{args.save_dir}/{args.csv_name}'
    

    if args.aac_svr:
        
        # Use simple AAC-SVR model instead of EpHod language model
        if args.verbose:
            print('Predicting with AAC-SVR model')
        all_ypred = predict_aac_svr(args, accessions, sequences)
    
    else:
        
        # Use EpHod deep language model        
        # Attention weights in output directory
        if args.save_attention_weights:
            attention_dir = f'{args.save_dir}/attention_weights'
            if not os.path.exists(attention_dir):
                os.makedirs(attention_dir)

        # EpHod embeddings file
        embed_file = f'{args.save_dir}/embeddings.csv'
    
        # Predict pHopt for sequences in batches
        # First, initialize EpHod model class with optimal learned weights
        ephod_model = models.EpHodModel()
        if args.verbose:
            print('Initializing EpHod model')
            print(f'Device is {ephod_model.device}')

        # Batch prediction
        num_batches = int(np.ceil(numseqs / args.batch_size))
        all_ypred, all_emb_ephod = [], []
        
        with torch.no_grad():
            batches = range(1, num_batches + 1)
            if args.verbose:
                batches = tqdm.tqdm(batches, desc="Predicting pHopt")
            
            for batch_step in batches:
                
                # Batch sequences
                start_idx = (batch_step - 1) * args.batch_size
                stop_idx = batch_step * args.batch_size
                accs = accessions[start_idx : stop_idx] 
                seqs = sequences[start_idx : stop_idx]
                
                # Predict with EpHod model
                ypred, emb_ephod, attention_weights = ephod_model.batch_predict(accs, seqs)
                all_ypred.extend(ypred.to('cpu').detach().numpy())
                all_emb_ephod.extend(emb_ephod.to('cpu').detach().numpy())
                if args.save_attention_weights:
                    _ = write_attention_weights(args, accs, seqs, attention_weights,
                                                attention_dir)
                
        if args.save_embeddings:
            all_emb_ephod = pd.DataFrame(np.array(all_emb_ephod), index=accessions)
            all_emb_ephod.to_csv(embed_file)

    if args.verbose:
        print('Prediction completed.')
        
        
    # Save predictions
    all_ypred = pd.DataFrame(all_ypred, index=accessions, columns=['pHopt'])
    all_ypred.to_csv(phout_file)
    
    
    




if __name__ == '__main__':
    main()
    
        
        
        
        



    
