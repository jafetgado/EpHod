"""
Feed-forward neural network
"""




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import torchActivation








class FeedForwardNetwork(nn.Module):
    '''A feed-forward neural network (fully-connected)'''
    
    def __init__(self,
                 input_dim=1280,
                 hidden_dims=[512,512],
                 dropout=0.25,
                 activation='relu', 
                 residual=False, 
                 random_seed=0):
        
        super(FeedForwardNetwork, self).__init__()
        
        # Model hyperparameters
        self.input_dim = input_dim   
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.dropout = dropout
        self.activation = activation         
        self.random_seed = random_seed       
        self.residual = residual
        if self.residual:
            assert self.num_layers > 1, \
                'Cannot have residual connections with less than 2 hidden layers'
            assert len(set(hidden_dims)) == 1, \
                'Cannot have different sizes of dense layers with residual=True'
        _ = torch.manual_seed(self.random_seed)
        

        # First dense layer
        # Separate first dense layer from the other dense layers to allow a residual 
        # connection between the output of the first and the last dense hidden layers.            
        self.first_dense = nn.ModuleList(
            [
                nn.Linear(self.input_dim, self.hidden_dims[0]),
                nn.BatchNorm1d(self.hidden_dims[0]),                                         
                torchActivation(self.activation),
                nn.Dropout(self.dropout) if self.activation != 'selu' \
                    else nn.AlphaDropout(self.dropout)
                    ]
                )
        
        # Other dense layers after first dense and residual connection
        if self.num_layers > 1:
            
            self.other_dense = nn.ModuleList()
            
            for i in range(1, self.num_layers):
                
                self.other_dense.extend(
                    [
                        nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]),
                        nn.BatchNorm1d(self.hidden_dims[i]),
                        torchActivation(self.activation),
                        nn.Dropout(self.dropout) if self.activation != 'selu' \
                            else nn.AlphaDropout(self.dropout)
                            ]
                        )
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dims[-1], 1)




    def forward(self, x):
        '''Forward pass of model'''

        # First dense layer
        for layer in self.first_dense:
            x = layer(x) 

        # Other dense layers
        if self.num_layers > 1:
            x0 = x
            
            for layer in self.other_dense:
                x = layer(x)
                
            # Skip/residual connection (shape is [batch_size, dense_dim])
            if self.residual:
                x = x0 + x 

        # Output layer
        y = self.output_layer(x)
        y = y.flatten()
        
        return y, x
    