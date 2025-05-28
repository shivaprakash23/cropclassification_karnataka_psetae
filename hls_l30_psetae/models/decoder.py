"""
Decoder module for HLS L30 data classification
"""

import torch.nn as nn


def get_decoder(n_neurons):
    """
    Create a decoder network for classification
    
    Args:
        n_neurons (list): List of neuron counts for each layer of the decoder
        
    Returns:
        nn.Sequential: The decoder network
    """
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        if i < (len(n_neurons) - 2):
            layers.extend([
                nn.BatchNorm1d(n_neurons[i + 1]),
                nn.ReLU()
            ])
    m = nn.Sequential(*layers)
    return m
