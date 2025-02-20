import torch
from torch import nn
import torch.nn.functional as F

from transformer_test import my_cross_att, my_mlp
from vae import Vae_Encoder

class Encoder_FPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln_1 = nn.LayerNorm([3, 64, 512])

        self.con_1x1 = nn.Conv2d(3, 3, 1, (1, 2))

        # 尺度一
        # 原始输入就是 尺度一

        # 尺度二
        self.conv_1_1 = nn.Conv2d(3, 3, (1, 3), stride=(1, 2), padding=(0, 1))
        self.conv_1_2 = nn.Conv2d(3, 3, (1, 3), stride=(1, 4), padding=(0, 1))

        # 尺寸三
        self.layer_2_1 = nn.Conv2d(3, 3, (3, 1), stride=(2, 2), padding=(1, 0))
        self.layer_2_2 = nn.Conv2d(3, 3, (3, 1), stride=(2, 1), padding=(1, 0))
        self.layer_2_3 = nn.Sequential(
            nn.ConvTranspose2d(3, 3, (1, 2), (1, 2), padding=0),
            nn.Conv2d(3, 3, (3, 1), stride=(2, 1), padding=(1, 0))
        )

        # 后续处理
        self.leaky_relu = nn.LeakyReLU()
        self.ada_ave_pool = nn.AdaptiveAvgPool2d((32, 256))

        self.linear_fa = nn.Linear(256, 128)
        self.linear_fb = nn.Linear(256, 128)
        self.linear_fc= nn.Linear(256, 128)

        self.relu = nn.ReLU()
        
    def forward(self, input):
        # torch.Size([3, 64, 512])
        input = self.ln_1(input)

        # for 残差
        # [3, 32, 256]
        res_in_out = self.con_1x1(input)
        res_in_out = self.ada_ave_pool(res_in_out)
        
        out_512 = input.clone().detach()  # torch.Size([3, 64, 512])
        out_256 = self.conv_1_1(input)    # torch.Size([3, 64, 256])
        out_128 = self.conv_1_2(input)    # torch.Size([3, 64, 128])

        # torch.Size([3, 64, 512])
        out_512 = self.layer_2_1(out_512) # torch.Size([3, 32, 256])
        out_256 = self.layer_2_2(out_256) # torch.Size([3, 32, 256])
        out_128 = self.layer_2_3(out_128) # torch.Size([3, 32, 256])

        out_a = self.leaky_relu(out_512)
        out_b = self.leaky_relu(out_256)
        out_c = self.leaky_relu(out_128)

        # return out_a, out_b, out_c

        # out_a = self.ada_ave_pool(out_a)
        # out_b = self.ada_ave_pool(out_b)
        # out_c = self.ada_ave_pool(out_c)

        out_a = out_a + res_in_out
        out_b = out_b + res_in_out
        out_c = out_c + res_in_out

        feature_a = out_a[0, :, :] + out_b[0, :, :] + out_c[0, :, :] 
        feature_b = out_a[1, :, :] + out_b[1, :, :] + out_c[1, :, :] 
        feature_c = out_a[2, :, :] + out_b[2, :, :] + out_c[2, :, :] 

        fa_project = self.relu(self.linear_fa(feature_a))
        fb_project = self.relu(self.linear_fb(feature_b))
        fc_project = self.relu(self.linear_fc(feature_c))

        return fa_project, fb_project, fc_project

class Encoder_Att(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = 64
        self.w = 512
        self.head_num = 4
        self.head_dim = self.w // self.head_num
        self.ln = nn.LayerNorm([1, self.h, self.w])
        self.W_q = nn.Linear(self.w, self.w)
        self.W_k = nn.Linear(self.w, self.w)
        self.W_v = nn.Linear(self.w, self.w)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.w, self.w),
            nn.ReLU(),
            nn.Linear(self.w, self.w // 2),
            nn.Conv2d(1, 1, 1, (2, 2))
        )

    def forward(self, input):
        input = self.ln(input)
        batch_size, seq_length, embed_dim = input.size()
        Q = self.W_q(input).view(batch_size, seq_length, self.head_num, self.head_dim).transpose(1, 2)
        K = self.W_k(input).view(batch_size, seq_length, self.head_num, self.head_dim).transpose(1, 2)
        V = self.W_v(input).view(batch_size, seq_length, self.head_num, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_num ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        res = self.ln(input + attention_output)
        res = self.ffn(res)
        return res

class My_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn = Encoder_FPN()
        self.att = Encoder_Att()
        
    def forward(self, input):
        # input = [4, 64, 512]
        fpn_in = input[0:3, :, :]   # [3, 64, 512]
        att_in = input[3:, :, :]    # [1, 64, 512]
        fa, fb, ff = self.fpn(fpn_in)
        fs = self.att(att_in)
        return fa.reshape(-1, 32, 128), fb.reshape(-1, 32, 128), ff.reshape(-1, 32, 128), fs

class My_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_att_1 = my_cross_att()
        self.cross_att_2 = my_cross_att()
        self.my_mlp_0 = my_mlp()
        self.ln = nn.LayerNorm([1, 32, 128])
        self.my_mlp_1 = my_mlp()
        self.my_mlp_2 = my_mlp()
        self.my_mlp_3 = my_mlp()
        self.my_mlp_4 = my_mlp()

    def forward(self, fa, fb, ff, fs):
        x = self.cross_att_1(fa, fb, ff)
        x = x + fs
        x = self.ln(x)
        x = self.my_mlp_0(x)
        x = self.cross_att_2(x, x, fs)
        x = self.ln(x)
        x1 = self.my_mlp_1(x) + fs
        x2 = self.my_mlp_2(x) + fs
        x3 = self.my_mlp_3(x) + fs
        x4 = self.my_mlp_4(x)
        return x1, x2, x3, x4

class My_Modal(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = Vae_Encoder()
        self.my_encoder = My_Encoder()
        self.my_decoder = My_Decoder()
    
    def forward(self, input):
        x = self.vae(input)
        fa, fb, ff, fs = self.my_encoder(x)
        x1, x2, x3, x4 = self.my_decoder(fa, fb, ff, fs)
        return x1, x2, x3, x4

if __name__ == '__main__':
    input = torch.randn((4, 64, 16384),requires_grad=False)
    my_modal = My_Modal()
    my_modal = my_modal.eval()

    x1, x2, x3, x4 = my_modal(input)

    torch.onnx.export(my_modal, input, "my_modal.onnx",opset_version=11)

    # all in [1, 32, 128]
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)

