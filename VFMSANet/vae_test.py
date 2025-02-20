import torch
from torch import nn


'''
100MHz * 1ms = 100000
[4, 64, 5000]
'''
class VAE_TEST(nn.Module):
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


if __name__ =='__main__':
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    VAE_encoder = VAE_TEST()
    VAE_encoder = VAE_encoder.to(device)
    input = torch.randn((4, 64, 16384)).to(device=device)
    for _ in range(500):
        output = VAE_encoder.encode(input)
    print(output.shape)
    output = VAE_encoder.decode(output)
    print(output.shape)

