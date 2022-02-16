import torch.nn as nn
import torch.nn.functional as F

class classification(nn.Module):
    def __init__(self, layer_size, activation=False):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(28*28, 16))
        for k in range(layer_size - 1):
            self.layers.append(nn.Linear(16, 16))
            # Output layer
        self.out = nn.Linear(16, 10)
      #  self.classifier = nn.Sequential(nn.Linear(in_features=28*28, out_features=10))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.activation is True:
            for layer in self.layers:
                x = F.leaky_relu(layer(x))
        else:
            for layer in self.layers:
                x = layer(x)
        x = self.out(x)
        #x = self.classifier(x)
        return x
