"""
Convolution neural network
"""




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import torchActivation
from models.fnn import FeedForwardNetWork








class ConvNet(nn.Module):

    def __init__(self,
                 input_channel=20,
                 input_length=1024,
                 start_conv_channel=32,
                 kernel_size=3,
                 num_conv_layers=8,                 
                 conv_dropout=0.25,
                 pooltype='max',
                 dense_dim=1024,
                 num_dense_layers=8,
                 dense_dropout=0.25,
                 activation='relu', 
                 random_seed=0):
        
        super(ConvNet, self).__init__()
        _ = torch.manual_seed(random_seed)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList() 
        for i in range(num_conv_layers):
            in_channel = input_channel if i==0 else out_channel
            out_channel = start_conv_channel * (2 ** i)
            self.conv_layers.append(nn.Conv1d(in_channel, 
                                              out_channel,
                                              kernel_size=kernel_size, 
                                              padding=(kernel_size // 2))
                                    )
            self.conv_layers.append(nn.BatchNorm1d(out_channel))
            self.conv_layers.append(torchActivation(activation))
            if pooltype=='max' and i>0:
                self.conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            elif pooltype=='average' and i>0:
                self.conv_layers.append(nn.AvgPool1d(kernel_size=2, stride=2))
            self.conv_layers.append(nn.Dropout(conv_dropout))
        
        
        # Residual dense layers
        self.flatsize = start_conv_channel * input_length
        self.residual_dense = FeedForwardNetwork(input_dim=self.flatsize, 
                                                 hidden_dims=[dense_dim] * num_dense_layers,
                                                 dropout=dense_dropout,
                                                 activation=activation,
                                                 residual=True, 
                                                 random_seed=random_seed)


    def forward(self, x):

        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(-1, self.flatsize)
        y = self.residual_dense(x)
        
        return y
