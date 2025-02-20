import torch
from torch import nn
import torch.nn.functional as F

class encode_att(nn.Module):
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

class my_cross_att(nn.Module):
    def __init__(self, channel_num=32, sample_num=128, head_num=4):
        super().__init__()
        self.channel_num = channel_num
        self.sample_num = sample_num
        self.head_num = head_num
        self.head_dim = self.sample_num // self.head_num

        # network config
        self.ln = nn.LayerNorm([1, self.channel_num, self.sample_num])
        self.W_q = nn.Linear(self.sample_num, self.sample_num)
        self.W_k = nn.Linear(self.sample_num, self.sample_num)
        self.W_v = nn.Linear(self.sample_num, self.sample_num)

    def forward(self, q_in, k_in, v_in):
        batch_size, seq_length, embed_dim = q_in.size()
        Q = self.W_q(q_in).view(batch_size, seq_length, self.head_num, self.head_dim).transpose(1, 2)
        K = self.W_k(k_in).view(batch_size, seq_length, self.head_num, self.head_dim).transpose(1, 2)
        V = self.W_v(v_in).view(batch_size, seq_length, self.head_num, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_num ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        return attention_output

class my_mlp(nn.Module):
    def __init__(self, in_feature=128, out_feature=128):
        super().__init__()
        self.ln = nn.LayerNorm([1, 32, 128])
        self.linear1 = nn.Linear(32, 32)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(in_feature, out_feature)

    def forward(self, input):
        input = self.ln(input)
        t = input.transpose(-1, -2)
        x = self.linear1(t)
        x = self.gelu(x)
        x = x.transpose(-1, -2)
        x = x + input
        
        x = self.ln(x)
        xx = self.linear2(x)
        xx = self.gelu(xx)
        xx = xx + x
        return xx
