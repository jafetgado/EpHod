"""Modular functions/classes for building and running EpHod model
Author: Japheth Gado
"""




import subprocess
import os
import builtins

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import esm

import ephod.utils as utils
import requests








def print(*args, **kwargs):
    '''Custom print function to always flush output when verbose'''

    builtins.print(*args, **kwargs, flush=True)
    
    






def download_models(get_from='zenodo'):
    '''Download saved models (EpHod and AAC-SVR)'''
    
    if get_from == 'googledrive':
        
        # Download from Google drive
        glink = "https://drive.google.com/drive/folders/138cnx4hFrzNODGK6A_yd9wo7WupKpSjI?usp=share_link/"
        cmd = f"gdown --folder {glink}"
        print('Downloading EpHod models from Google drive with gdown\n')
        _ = subprocess.call(cmd, shell=True) # Download model from google drive 

    elif get_from == 'zenodo':
        
        # Download from Zenodo
        zlink = "https://zenodo.org/record/8011249/files/saved_models.tar.gz?download=1"
        print('Downloading EpHod models from Zenodo with requests\n')
        response = requests.get(zlink, stream=True)
        if response.status_code == 200:
            with requests.get(zlink, stream=True) as r:
                r.raise_for_status()
                with open("saved_models.tar.gz", 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
        else:
            print(f"Request failed with status code {response.status_code}")

    
    else: 
        raise ValueError(f"Value of get_from ({get_from}) must be 'googledrive' or 'zenodo'")
        
    
    # Move downloaded models to proper location
    this_dir, this_filename = os.path.split(__file__)
    if get_from == 'zenodo':
        # Untar downloaded file
        tarfile = os.path.join(this_dir, 'saved_models') 
        _ = subprocess.call(f"tar -xvzf {tarfile}", shell=True)
        _ = subprocess.call(f"rm -rfv {tarfile}", shell=True)
    
    save_path = os.path.join(this_dir, 'saved_models') 
    cmd = f"mv -f ./saved_models {save_path}/"
    print(cmd)
    print(f'\nMoving downloaded models to {save_path}')
    _ = subprocess.call(cmd, shell=True)
    error_msg = "RLAT model failed to download!"
    assert os.path.exists(f"{save_path}/RLAT/RLAT.pt"), error_msg
    
    
    
    
    
    
    
    
def torchActivation(activation='elu'):
    '''Return an activation function from torch.nn'''

    if activation == 'relu':
        return nn.ReLU()

    elif activation == 'leaky_relu':
        return nn.LeakyReLU()

    elif activation == 'elu':
        return nn.ELU()

    elif activation == 'selu':
        return nn.SELU()
    
    elif activation == 'gelu':
        return nn.GELU()
        







def count_parameters(model):
    '''Return a count of parameters and tensor shape of PyTorch model''' 
    
    counted = {}
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total += count
            counted[name] = count
    counted['FULL_MODEL'] = total

    return counted
    







class ResidualDense(nn.Module):
    '''A single dense layer with residual connection'''
    
    def __init__(self, dim=2560, dropout=0.1, activation='elu', random_seed=0):
        
        super(ResidualDense, self).__init__()
        _ = torch.manual_seed(random_seed)
        self.dense = nn.Linear(dim, dim)
        self.batchnorm = nn.BatchNorm1d(dim)
        self.activation = torchActivation(activation)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        x0 = x
        x = self.dense(x)
        x = self.batchnorm(x)
        x = self.activation(x)        
        x = self.dropout(x)
        x = x0 + x
        
        return x








class LightAttention(nn.Module):
    '''Convolution model with attention to learn pooled representations from embeddings'''

    def __init__(self, dim=1280, kernel_size=7, random_seed=0):
        
        super(LightAttention, self).__init__()
        _ = torch.manual_seed(random_seed)        
        samepad = kernel_size // 2
        self.values_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=samepad)
        self.weights_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=samepad)
        self.softmax = nn.Softmax(dim=-1)
    
    
    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.int32)  # Don't mask out
        values = self.values_conv(x)
        values = values.masked_fill(mask[:,None,:]==0, -1e6)
        weights = self.weights_conv(x)
        weights = weights.masked_fill(mask[:,None,:]==0, -1e6)
        weights = self.softmax(weights)
        x_sum = torch.sum(values * weights, dim=-1) # Attention-weighted pooling
        x_max, _ = torch.max(values, dim=-1) # Max pooling
        x = torch.cat([x_sum, x_max], dim=1)
        
        return x, weights
    
    
    





class ResidualLightAttention(nn.Module):
    '''Model consisting of light attention followed by residual dense layers'''
    
    def __init__(self, dim=1280, kernel_size=9, dropout=0.5,
                 activation='relu', res_blocks=4, random_seed=0):

        super(ResidualLightAttention, self).__init__()
        torch.manual_seed(random_seed)
        self.light_attention = LightAttention(dim, kernel_size, random_seed)
        self.batchnorm = nn.BatchNorm1d(2 * dim)                
        self.dropout = nn.Dropout(dropout)        
        self.residual_dense = nn.ModuleList()        
        for i in range(res_blocks):
            self.residual_dense.append(
                ResidualDense(2 * dim, dropout, activation, random_seed)
                )
        self.output = nn.Linear(2 * dim, 1)
        
        
    def forward(self, x, mask=None):

        x, weights = self.light_attention(x, mask)
        x = self.batchnorm(x)
        x = self.dropout(x)
        for layer in self.residual_dense:
            x = layer(x)
        y = self.output(x).flatten()
        
        return [y, x, weights]








class EpHodModel():
    
    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device != 'cuda':
            print('WARNING: You are not using a GPU which will be slow.')
        self.esm1v_model, self.esm1v_batch_converter = self.load_ESM1v_model()
        self.rlat_model = self.load_RLAT_model()
        _ = self.esm1v_model.eval()
        _ = self.rlat_model.eval()
        
    
    def load_ESM1v_model(self):
        '''Return pretrained ESM1v model weights and batch converter'''
        
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
        model = model.to(self.device)
        batch_converter = alphabet.get_batch_converter()
        
        return model, batch_converter
    
    
    def get_ESM1v_embeddings(self, accs, seqs):
        '''Return per-residue embeddings (padded) for protein sequences from ESM1v model'''

        seqs = [utils.replace_noncanonical(seq, 'X') for seq in seqs]
        data = [(accs[i], seqs[i]) for i in range(len(accs))]
        batch_labels, batch_strs, batch_tokens = self.esm1v_batch_converter(data)
        batch_tokens = batch_tokens.to(device=self.device, non_blocking=True)
        emb = self.esm1v_model(batch_tokens, repr_layers=[33], return_contacts=False)
        emb = emb["representations"][33]
        emb = emb.transpose(2,1) # From (batch, seqlen, features) to (batch, features, seqlen)
        emb = emb.to(self.device)

        return emb
    
    
    def load_RLAT_model(self):
        '''Return fine-tuned residual light attention top model'''

        # Path to RLAT model
        this_dir, this_filename = os.path.split(__file__)
        params_path = os.path.join(this_dir, 'saved_models', 'RLAT', 'params.json')
        rlat_path = os.path.join(this_dir, 'saved_models', 'RLAT', 'RLAT.pt')
        
        # Download RLAT model from google drive if not in path
        if not os.path.exists(rlat_path):
            _ = download_models()
        
        # Load RLAT model from path
        checkpoint = torch.load(rlat_path, map_location=self.device)
        params = utils.read_json(params_path)        
        model = ResidualLightAttention(**params)
        model = DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model = model.to(self.device)

        return model

    
    def batch_predict(self, accs, seqs):
        '''Predict pHopt with EpHod on a batch of sequences'''
        
        emb_esm1v = self.get_ESM1v_embeddings(accs, seqs)
        maxlen = emb_esm1v.shape[-1]
        masks = [[1] * len(seqs[i]) + [0] * (maxlen - len(seqs[i])) \
                 for i in range(len(seqs))]
        masks = torch.tensor(masks, dtype=torch.int32)
        masks = masks.to(self.device)
        output = self.rlat_model(emb_esm1v, masks) # (ypred, emb_ephod, attention_weights)
        
        return output
    
    
    

