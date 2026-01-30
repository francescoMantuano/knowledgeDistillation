import torch.nn as nn

#classe per effettuare la proiezione lineare siccome teacher e student hanno dimensioni differenti
class FeatureProjector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.proj(x)
    

