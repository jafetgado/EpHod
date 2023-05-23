"""
Models based on light-attention architecture (LAT, PAT, DCAT)
"""




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import torchActivation
from models.fnn import FeedForwardNetwork









class AttentionTopModel(nn.Module):
    '''
    Class to implement a top model based on LAT architecture to predict a property from 
    embeddings.
    
        

    Parameters
    -----------
    input_dim : int
        Channel dimension of input embedding, 1280 for ESM-1b        
    attention_dim : int
        Channel dimension of transformed values and attention weights
    kernel_size : int
        Kernel size of convolution 
    conv_dropout : float
        Dropout applied to values embedding after convolution 
    perceptive : bool
        If True, concatenate input embedding with average pooled embedding to derive 
        a perceptive embedding.        
    dense_dim : int
        Number of units in hidden dense layer
    dense_dropout : float
        Dropout of dense hidden layers.
    activation : str
        Activation function of dense hidden layer.
    random_seed : int
        Random seed for reproducibillity
     
     '''


    def __init__(self, 
                 input_dim=1280,
                 attention_dim=1280,
                 kernel_size=9,
                 conv_dropout=0.25,
                 perceptive=True,                 
                 dense_dim=512,
                 dense_dropout=0.25,
                 num_layers=1, 
                 residual=False,
                 activation='relu',
                 random_seed=0):
        

        super(AttentionTopModel, self).__init__()


        # Initialize class parameters  
        self.input_dim = input_dim            
        self.attention_dim = attention_dim   
        self.kernel_size = kernel_size       
        self.conv_dropout = conv_dropout    
        self.perceptive = perceptive                 
        self.dense_dim = dense_dim           
        self.dense_dropout = dense_dropout   
        self.num_layers = num_layers
        self.residual = residual
        self.activation = activation         
        self.random_seed = random_seed       
        _ = torch.manual_seed(self.random_seed)

            
        # A convolution layer to learn values from embeddings
        self.values_conv = nn.Conv1d(
            2 * self.input_dim if perceptive else self.input_dim,
            self.attention_dim, 
            kernel_size=self.kernel_size,
            stride=1,
            padding=(self.kernel_size // 2)
            )
        self.values_dropout = nn.Dropout(self.conv_dropout) 
        
        
        # A convolution and softmax layer to learn attention weights that sum to 1
        self.attn_conv = nn.Conv1d(
            2 * self.input_dim if perceptive else self.input_dim,
            self.attention_dim, 
            kernel_size=self.kernel_size,
            stride=1,
            padding=(self.kernel_size // 2)
            )
        self.attn_softmax = nn.Softmax(dim=-1)
        
        # Dense(residual) layers on top of the attention-learned embeddings
        self.dense_layers = FeedForwardNetwork(
            input_dim=(2 * self.attention_dim), 
            hidden_dims=[self.dense_dim] * self.num_layers, 
            dropout=self.dense_dropout, 
            activation=self.activation, 
            residual=self.residual, 
            random_seed=self.random_seed
            )
                                                     


    def forward(self, x, mask=None):
        '''
        Forward pass of network. 

        Parameters
        ------------
        x : torch.Tensor
            Shape is [batch_size, embeddings_dim, seq_len]
        mask : torch.Tensor or None
            Boolean tensor indicating padded (0) and non-padded (1) positions. Shape is 
            [batch_size, seq_len]. If mask is None, a tensor of ones is used, 
            indicating that padded positions are not masked out in the forward 
            computations.

        Returns
        -------
        output: list
            A list of predicted values (shape is [batch_size, 1]), embedding of the
            hidden layer before the output layer (shape is [batch_size, dense_dim]), and
            the masked attention weights (shape is [batch_size, attention_dim, seq_len])
        '''
        
        # Masking for  padded positions
        if mask is None:
            # If mask is None, use all positions including padded positions
            mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.int32)
        
        # Derive a perceptive tensor by concatenating mean pooled embeddings to the input 
        # Output shape of perceptive tensor is [batch_size, 2 * input_dim, seq_length]
        if self.perceptive:
            # Mask out padded positions
            x_perc = x.masked_fill(mask[:,None,:] == 0, 0)  
            # Average pooling over unpadded positions
            x_perc = torch.divide(torch.sum(x_perc, dim=-1), 
                                  torch.sum(mask, dim=1)[:,None]) 
            # Repeat pooled tensor to match sequence length
            x_perc = x_perc[:,:,None].repeat(1,1,x.shape[-1]) 
            # Concatenate input embeddings and perceptive tensor along feature dimension
            x = torch.cat((x, x_perc), axis=1) 
        
        # Derive values and attention weights from input embeddings with convolution
        # Output shapes for values/weights is [batch_size, attention_dim, seq_length]
        values = self.values_conv(x)
        values = self.values_dropout(values)
        values = values.masked_fill(mask[:,None,:] == 0, -1e6)  # Mask out for max pooling

        # Attention weights that sum to 1
        attn = self.attn_conv(x)   
        attn = attn.masked_fill(mask[:,None,:] == 0, -1e6) # Mask out for average pooling
        attn = self.attn_softmax(attn)
        
        # Derive attention-learned embeddings by multiplying values and attention weights
        # Output shape is [batch_size, 2 * attention_dim]
        z_sum = torch.sum(values * attn, dim=-1) 
        z_max, _ = torch.max(values, dim=-1)
        z = torch.cat([z_sum, z_max], dim=1)

        # Dense layers
        y, z = self.dense_layers(z)

        
        return [y, z, attn]









class DilatedConvTopModel(nn.Module):

    def __init__(self, 
                 input_dim=1280,
                 input_len=1024,
                 attention_kernel_size=9,                 
                 conv_dim=64,
                 conv_kernel_size=3,
                 conv_dropout=0.1,
                 dilated_layers=9,                 
                 dense_dim=512,
                 dense_layers=1,                  
                 dense_dropout=0.25,
                 activation='relu',
                 random_seed=0):
        
        super(DilatedConvTopModel, self).__init__()
        _ = torch.manual_seed(random_seed)
        
        
        # A convolution layer to learn transformed values from embeddings
        self.values_conv_layer = nn.Conv1d(
            input_dim,
            conv_dim, 
            kernel_size=attention_kernel_size,
            stride=1, 
            dilation=1, 
            padding=(attention_kernel_size // 2)
            )        
        
        
        # A convolution and softmax layer to learn attention weights that sum to 1
        # Attention-weighted values are obtained by element-wise multiplication with weights
        self.attention_conv_layer = nn.Conv1d(
            input_dim,
            conv_dim, 
            kernel_size=attention_kernel_size,
            stride=1, 
            dilation=1, 
            padding=(attention_kernel_size // 2)
            )       
        self.attention_softmax = nn.Softmax(dim=-1)
        self.conv_dropout_layer = nn.Dropout(conv_dropout) if activation != 'selu' \
                    else nn.AlphaDropout(conv_dropout)
        
        
        # Convolutional layers with increasing dilation rates
        self.dilated_conv_layers = nn.ModuleList()
        dilations = np.power(2, np.arange(dilated_layers)) 

        for d in dilations:

            same_padding = self.get_same_padding_size(conv_kernel_size, d)
            self.dilated_conv_layers.append(nn.Conv1d(conv_dim, 
                                                      conv_dim,
                                                      kernel_size=conv_kernel_size,
                                                      padding=same_padding,
                                                      dilation=d)
                              )
            self.dilated_conv_layers.append(nn.BatchNorm1d(conv_dim))
            self.dilated_conv_layers.append(torchActivation(activation))
            
        
        # Dense layers
        self.dense_layers = FeedForwardNetwork(input_dim=(input_len * conv_dim), 
                                               hidden_dims=[dense_dim] * dense_layers, 
                                               dropout=dense_dropout, 
                                               activation=activation, 
                                               residual=(dense_layers > 1), 
                                               random_seed=random_seed)

        
        
        

    def get_same_padding_size(self, k, d):

        padsize = int(
            np.ceil(
                d * (k - 1) / 2
                )
            )
        
        return padsize
    
    
    
    
    def forward(self, x, mask=None):
        
        # Masking for padded positions
        if mask is None:
            # If mask is None, use all positions including padded positions
            mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.int32)
        
        # Derive attention-weighted values from input embeddings
        # Shapes for values/weights is [batch_size, attention_dim, seq_length]
        x_values = self.values_conv_layer(x)
        x_weights = self.attention_conv_layer(x)   
        x_weights = x_weights.masked_fill(mask[:,None,:] == 0, -1e6) 
        x_weights = self.attention_softmax(x_weights) # Weights sum to 1
        x = (x_values * x_weights)  
        x = self.conv_dropout_layer(x)
        
        # Dilated convolution layers
        x0 = x
        for layer in self.dilated_conv_layers:
            x = layer(x)
        x = x0 + x  # Residual connection
        
        # Dense layers
        x = x.flatten(start_dim=1, end_dim=2)
        y, z = self.dense_layers(x)

        
        return [y, z, x_weights]     








    
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
                   
    