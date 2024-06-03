import torch.nn as nn
import torch

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hide1_num, hide2_num, dropout1, dropout2, output_dim):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hide1_num),  
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hide1_num, hide2_num), 
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hide2_num, output_dim)
        )
        self.layers.apply(self.init_weights)
        
    def forward(self, x):
        return self.layers(x)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
