'''
500MHz, 1ms = 500000样本
50MHz, 1ms = 50000样本
[4, 64, 50000]
'''

import torch
from torch import nn

class Vae_Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    

        self.layer_config = [16384, 8192, 4096, 2048, 1024, 512]

        '''
        '''
        self.encoder = nn.Sequential(
            nn.Linear(self.layer_config[0], self.layer_config[1]),
            nn.LeakyReLU(),
            nn.Linear(self.layer_config[1], self.layer_config[2]),
            nn.ReLU(),
            nn.Linear(self.layer_config[2], self.layer_config[3]),
            nn.LeakyReLU(),
            nn.Linear(self.layer_config[3], self.layer_config[4]),
            nn.ReLU(),
            nn.Linear(self.layer_config[4], self.layer_config[5]),
        )

        '''
        '''
        self.decoder = nn.Sequential(            
            nn.Linear(self.layer_config[5], self.layer_config[4]),
            nn.LeakyReLU(),
            nn.Linear(self.layer_config[4], self.layer_config[3]),            
            nn.ReLU(),
            nn.Linear(self.layer_config[3], self.layer_config[2]),
            nn.LeakyReLU(),
            nn.Linear( self.layer_config[2],self.layer_config[1]),
            nn.ReLU(),
            nn.Linear(self.layer_config[1], self.layer_config[0]),
        )

    '''
    input: [4, 64, 12800]
    output: [4, 64, 400]
    '''
    def encode(self, input):
        return self.encoder(input)
    def decode(self, input):
        return self.decoder(input)
    def forward(self, input):
        return self.encoder(input)


if __name__ == '__main__':
    my_vae = Vae_Encoder()
    input = torch.randn((4, 64, 16384))
    print(my_vae(input).shape)