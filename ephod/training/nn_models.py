"""
Neural network models
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




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




class FeedForwardNetwork(nn.Module):
    '''A feed-forward neural network (fully-connected)'''
    
    def __init__(self, input_dim=1280, hidden_dim=128, num_layers=1, dropout=0.25, activation='relu', 
                 residual=False):
        
        super().__init__()
        self.num_layers = num_layers
        self.residual = residual

        # First dense layer
        # Separate first dense layer from the other dense layers to allow residual connection
        self.first_dense = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),                                         
                torchActivation(activation),
                nn.Dropout(dropout) if activation != 'selu' else nn.AlphaDropout(dropout)
            ]
        )
        
        # Other dense layers after first dense and residual connection
        if num_layers > 1:
            self.other_dense = nn.ModuleList()
            for i in range(1, num_layers):
                self.other_dense.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        torchActivation(activation),
                        nn.Dropout(dropout) if activation != 'selu' else nn.AlphaDropout(dropout)
                    ]
                )
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)


    def forward(self, x):

        # First dense layer
        for layer in self.first_dense:
            x = layer(x) 
            
        # Other dense layers
        if self.num_layers > 1:
            x0 = x
            for layer in self.other_dense:
                x = layer(x)  
            # Residual connection
            if self.residual:
                x = x0 + x 
        # Output layer
        y = self.output_layer(x).flatten()
        
        return y




class CNN(nn.Module):

    def __init__(self, input_channel=20, input_length=1024, start_conv_channel=32, kernel_size=3,
                 num_conv_layers=8, conv_dropout=0.25, pooltype='max', dense_dim=128, num_dense_layers=1,
                 dense_dropout=0.25, activation='relu'):
        
        super().__init__()
        # Conv layers
        self.conv_layers = nn.ModuleList() 
        for i in range(num_conv_layers):
            in_channel = input_channel if i==0 else out_channel
            out_channel = start_conv_channel * (2 ** i)
            self.conv_layers.append(nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, 
                                              padding=(kernel_size // 2)))
            self.conv_layers.append(nn.BatchNorm1d(out_channel))
            self.conv_layers.append(torchActivation(activation))
            if pooltype=='max' and i>0:
                self.conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            elif pooltype=='average' and i>0:
                self.conv_layers.append(nn.AvgPool1d(kernel_size=2, stride=2))
            self.conv_layers.append(nn.Dropout(conv_dropout))
        # FNN layers
        self.flatsize = start_conv_channel * input_length
        self.fnn = FeedForwardNetwork(input_dim=self.flatsize, hidden_dim=dense_dim, num_layers=num_dense_layers, 
                                      dropout=dense_dropout, activation=activation, residual=True)

    def forward(self, x):

        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, self.flatsize)
        y = self.fnn(x)
        
        return y




class CausalConv1d(nn.Module):
    '''Causal convolution'''
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):

        super().__init__()

        self.pad = int(np.ceil(dilation * (kernel_size - 1) / 2)) # Padding to ensure causal convolution
        self.conv_layer = nn.Conv1d(in_channels, out_channels, padding=0,kernel_size=kernel_size, 
                                    dilation=dilation)
    
    def forward(self, x):
        
        len1 = x.shape[-1]
        x = F.pad(x, (self.pad * 2, 0))
        x = self.conv_layer(x)
        len2 = x.shape[-1]
        start = len2 - len1
        x = x[:,:,start:len2]

        return x




class DCNNBlock(nn.Module):

    def __init__(self,channel_size=32, num_layers=6,dropout=0.5, activation='elu'):
        
        super().__init__()
        dilation_rates = (2. ** np.arange(num_layers)).astype(np.int32)
        self.layers = nn.ModuleList()
        for d in dilation_rates:
            self.layers.append(CausalConv1d(channel_size, channel_size, kernel_size=2, dilation=d))
            self.layers.append(nn.BatchNorm1d(channel_size))
            self.layers.append(torchActivation(activation))
        self.layers.append(nn.Dropout(dropout))
        
    def forward(self, x):
        
        x0 = x
        for layer in self.layers:
            x = layer(x)
        x = x0 + x

        return x




class DCNN(nn.Module):

    def __init__(self, input_channel=20, conv_channel=48, num_blocks=6, layers_per_block=10,
                 block_dropout=0.5, dense_dim=128, num_dense_layers=2, dense_dropout=0.25,
                 activation='elu'):
        
        super().__init__()
        self.embed_layer = nn.Conv1d(input_channel, conv_channel, kernel_size=1, bias=False)
        
        # DCNN blocks
        self.dcnn_blocks = nn.ModuleList()    
        for block in range(num_blocks):
            self.dcnn_blocks.append(DCNNBlock(channel_size=conv_channel, num_layers=layers_per_block,
                                              dropout=block_dropout, activation=activation))
        
        self.fnn = FeedForwardNetwork(input_dim=conv_channel, hidden_dim=dense_dim, 
                                      num_layers=num_dense_layers, dropout=dense_dropout, 
                                      activation=activation, residual=True)
    
    def forward(self, x, mask=None):

        x = self.embed_layer(x)
        # Dilated convolutions
        for layer in self.dcnn_blocks:
            x = layer(x)
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.int32)
        # Average pooling
        x = x.masked_fill(mask[:,None,:] == 0, 0)
        x = torch.divide(torch.sum(x, dim=-1), torch.sum(mask, dim=-1)[:,None])
        
        # FNN
        y = self.fnn(x).flatten()
        
        return y




class RNN(nn.Module):

    def __init__(self, input_channel=20, input_length=1024, gru_dim=1024, conv_downsample=1, 
                 conv_dropout=0.25, dense_dim=128, dense_dropout=0.25, activation='relu'):
        
        super().__init__()
        
        # Convolutional layers to downsample
        self.conv_downsample = conv_downsample
        self.conv_layers = nn.ModuleList() 
        pad = (conv_downsample // 2 - 1)
        pad = max(pad, 0)
        self.conv_layers.append(nn.Conv1d(input_channel, gru_dim, kernel_size=conv_downsample, 
                                          stride=conv_downsample, padding=pad))
        self.conv_layers.append(nn.BatchNorm1d(gru_dim))
        self.conv_layers.append(torchActivation(activation))
        self.conv_layers.append(nn.Dropout(conv_dropout))
        
        # RNN layer
        self.gru_layer = nn.GRU(gru_dim, gru_dim, batch_first=True, bidirectional=False)
       
        # Dense layer
        self.dense_layers = nn.ModuleList()
        self.dense_layers.extend(
            [
                nn.Linear(gru_dim, dense_dim),
                nn.BatchNorm1d(dense_dim),                    
                torchActivation(activation),
                nn.Dropout(dense_dropout) if activation != 'selu' else nn.AlphaDropout(dense_dropout)
            ]
        )

        # Output layer
        self.output_layer = nn.Linear(dense_dim, 1) 



    def forward(self, x, mask=None):
        
        if mask is None:
            mask = torch.tensor(np.ones((x.shape[0], x.shape[-1])), dtype=torch.float32)

        # Downsample with convolution
        for layer in self.conv_layers:
            x = layer(x)

        # RNN
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

