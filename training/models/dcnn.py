"""
Convolution neural network
"""




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import torchActivation
from models.fnn import FeedForwardNetWork







class CausalConv1d(nn.Module):
    '''Causal convolution for autoregressive processing'''
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 dilation=1):

        super(CausalConv1d, self).__init__()

        # Top pad only, no bottom pad, to ensure causal convolution
        self.pad = int(np.ceil(dilation * (kernel_size - 1) / 2))
        self.conv_layer = nn.Conv1d(in_channels, 
                                    out_channels, 
                                    padding=0,
                                    kernel_size=kernel_size,
                                    dilation=dilation)

    
    def forward(self, x):
        
        # Causal padding: Pad the only the top of tensor, not bottom
        len1 = x.shape[-1]
        x = F.pad(x, (self.pad * 2, 0))
        
        # Convolution on padded tensor
        x = self.conv_layer(x)
        len2 = x.shape[-1]
        start = len2 - len1
        x = x[:,:,start:len2]

        return x







    
class ResidualBlockDCNN(nn.Module):

    def __init__(self,
                 channel_size=32,
                 num_layers=6,
                 dropout=0.5,
                 activation='elu'):
        super(ResidualBlockDCNN, self).__init__()

        dilation_rates = (2. ** np.arange(num_layers)).astype(np.int32)
        self.layers = nn.ModuleList()
        
        for d in dilation_rates:
            self.layers.append(CausalConv1d(channel_size, 
                                            channel_size, 
                                            kernel_size=2,
                                            dilation=d))
            self.layers.append(nn.BatchNorm1d(channel_size))
            self.layers.append(torchActivation(activation))
        self.layers.append(nn.Dropout(dropout))


    def forward(self, x):
        
        x0 = x
        for layer in self.layers:
            x = layer(x)
        x = x0 + x

        return x








class DilatedConvNet(nn.Module):

    def __init__(self,
                 input_channel=20,
                 conv_channel=48,
                 num_blocks=6,
                 layers_per_block=10,
                 block_dropout=0.5,
                 attention_dim=1024,
                 attention_kernel_size=9,
                 attention_dropout=0.25,
                 perceptive=True,
                 residual=True,
                 dense_dim=512,
                 num_dense_layers=4,
                 dense_dropout=0.25,
                 activation='elu', 
                 random_seed=0):
        
        super(DilatedConvNet, self).__init__()
        _ = torch.manual_seed(random_seed)
        
        # Embedding layer
        self.embed_layer = nn.Conv1d(input_channel, 
                                     conv_channel, 
                                     kernel_size=1, 
                                     bias=False)
        
        # Residual blocks of dilated convolutions
        self.residual_blocks = nn.ModuleList()    
        for block in range(num_blocks):
            self.residual_blocks.append(
                ResidualBlockDCNN(channel_size=conv_channel,
                               num_layers=layers_per_block,
                               dropout=block_dropout,
                               activation=activation))
        
        # Residual perceptive attention on top of 
        # Top dense model on average and max pooled tensor
        self.dense = FeedForwardNetwork(input_dim=(2 * conv_channel),
        								hidden_dims=[dense_dim] * num_dense_layers,
        								dropout=dense_dropout,
        								activation=activation,
        								residual=True, 
        								random_seed=random_seed)
        
        


    def forward(self, x, mask=None):

        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.int32)
        
        x = self.embed_layer(x)
        
        # Dilated convolutions
        for layer in self.residual_blocks:
            x = layer(x)
        
        # Average and max pooling
        xmax = x.masked_fill(mask[:,None,:] == 0, -1e8)
        xmax, _ = torch.max(xmax, dim=-1)
        xavg = x.masked_fill(mask[:,None,:] == 0, 0)
        xavg = torch.divide(torch.sum(xavg, dim=-1), 
                            torch.sum(mask, dim=-1)[:,None])
        x = torch.cat((xavg, xmax), axis=-1)
        
        
        # Dense Output
        y = self.dense(x).flatten()
        
        return y
