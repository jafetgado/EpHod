"""
Recurrent neural network (GRU)
"""




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import torchActivation









class GRUModel(nn.Module):

    def __init__(self,
                 input_channel=20,
                 input_length=1024,
                 gru_dim=1024,
                 conv_downsample=4, # 1, 2, or 4
                 conv_dropout=0.25,
                 dense_dim=1024,
                 dense_dropout=0.25,
                 activation='relu', 
                 random_seed=0):
        
        super(gruModel, self).__init__()
        _ = torch.manual_seed(random_seed)
        
        # Convolutional layers to downsample time/sequence dimesion
        self.conv_downsample = conv_downsample
        self.conv_layers = nn.ModuleList() 
        pad = (conv_downsample // 2 - 1)
        pad = max(pad, 0)
        self.conv_layers.append(nn.Conv1d(input_channel, 
                                          gru_dim,
                                          kernel_size=conv_downsample, 
                                          stride=conv_downsample,
                                          padding=pad)
                                )
        self.conv_layers.append(nn.BatchNorm1d(gru_dim))
        self.conv_layers.append(torchActivation(activation))
        self.conv_layers.append(nn.Dropout(conv_dropout))
    
        
        # Unidirectional GRU layer
        self.gru_layer = nn.GRU(gru_dim, 
                                  gru_dim, 
                                  batch_first=True, 
                                  bidirectional=False)
       

        # Dense layer
        self.dense_layers = nn.ModuleList()
        self.dense_layers.extend(
            [
                nn.Linear(gru_dim, dense_dim),
                nn.BatchNorm1d(dense_dim),                    
                torchActivation(activation),
                nn.Dropout(dense_dropout) if activation != 'selu' \
                    else nn.AlphaDropout(dense_dropout)
                    ]
                )

        # Output layer
        self.output_layer = nn.Linear(dense_dim, 1) 



    def forward(self, x, mask):
        
        
        if mask is None:
            mask = torch.tensor(np.ones((x.shape[0], x.shape[-1])), dtype=torch.float32)
        
        
        # Downsample with convolution
        for layer in self.conv_layers:
            x = layer(x)

        # gru
        x = x.transpose(2, 1) # Transpose from [batch, feat, seqlen] to [batch, sequence, features]
        x = self.gru_layer(x)[0]
        lastind = torch.sum(mask, axis=1, dtype=int) # index of last residue in sequence
        lastind = (lastind // self.conv_downsample) - 1  # index after conv. downsampling
        x = [x[i][lastind[i]] for i in range(len(lastind))] # get hidden state of last residue for each sequence
        x = torch.stack(x)
        
        # Dense layer
        for layer in self.dense_layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x).flatten()

        return x

