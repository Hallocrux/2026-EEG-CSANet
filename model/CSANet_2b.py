import numpy as np
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import time


def _rearrange(x, mode='to'):
    if mode == 'to':  # to (B, L, C)
        return x.transpose(-1, -2)  # [B, C, L] -> [B, L, C]
    else:  # to (B, C, L)
        return x.transpose(-1, -2)  # [B, L, C] -> [B, C, L]


class MSC(nn.Module):
    def __init__(self, dim, num_heads=8, topk=True, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., k1=2, k2=3):
        super(MSC, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.k1 = k1
        self.k2 = k2

        self.attn1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.avgpool1 = nn.AvgPool1d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool1d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool1d(kernel_size=kernel[2], stride=s[2], padding=pad[2])

        self.layer_norm = nn.LayerNorm(dim)
        self.topk = topk

    def forward(self, x, y):
        y1 = self.avgpool1(y)  # (B, 32, T)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        y = y1 + y2 + y3  # [B, C, L2]

        y = _rearrange(y, 'to')  # [B, L2, C]
        y = self.layer_norm(y)
        x = _rearrange(x, 'to')  # [B, L1, C]

        B, N1, C = y.shape
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # --- Top-k 1 ---
        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device)
        topk_indices1 = torch.topk(attn, k=max(1, N1 // self.k1), dim=-1, largest=True).indices
        mask1.scatter_(-1, topk_indices1, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = attn1 @ v

        # --- Top-k 2 ---
        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device)
        topk_indices2 = torch.topk(attn, k=max(1, N1 // self.k2), dim=-1, largest=True).indices
        mask2.scatter_(-1, topk_indices2, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = attn2 @ v

        out = out1 * self.attn1 + out2 * self.attn2
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = _rearrange(out, 'from')
        return out


class MSC_nok(nn.Module):
    def __init__(self, dim, num_heads=8, topk=True, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(MSC_nok, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.avgpool1 = nn.AvgPool1d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool1d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool1d(kernel_size=kernel[2], stride=s[2], padding=pad[2])

        self.layer_norm = nn.LayerNorm(dim)
        self.topk = topk

    def forward(self, x, y):
        y1 = self.avgpool1(y)  # (B, 32, T)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        y = y1 + y2 + y3  # [B, C, L2]

        y = _rearrange(y, 'to')  # [B, L2, C]
        y = self.layer_norm(y)
        x = _rearrange(x, 'to')  # [B, L1, C]

        B, N1, C = y.shape
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        out = _rearrange(out, 'from')
        return out


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        nn.init.kaiming_uniform_(self.conv1d.weight, nonlinearity='linear')

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv1d(x)


class TCN_block(nn.Module):
    def __init__(self, depth=2):
        super(TCN_block, self).__init__()
        self.depth = depth
        self.Activation_1 = nn.ELU()
        self.TCN_Residual_1 = nn.Sequential(
            CausalConv1d(32, 32, 4, dilation=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            CausalConv1d(32, 32, 4, dilation=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
        )
        self.TCN_Residual = nn.ModuleList()
        self.Activation = nn.ModuleList()
        for i in range(depth - 1):
            TCN_Residual_n = nn.Sequential(
                CausalConv1d(32, 32, 4, dilation=2 ** (i + 1)),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(0.3),
                CausalConv1d(32, 32, 4, dilation=2 ** (i + 1)),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(0.3),
            )
            self.TCN_Residual.append(TCN_Residual_n)
            self.Activation.append(nn.ELU())

    def forward(self, x):
        block = self.TCN_Residual_1(x)
        block += x
        block = self.Activation_1(block)
        for i in range(self.depth - 1):
            block_o = block
            block = self.TCN_Residual[i](block)
            block += block_o
            block = self.Activation[i](block)
        return block


class conv_block1(nn.Module):
    def __init__(self, ):
        super(conv_block1, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), bias=False, padding='same'),
            nn.BatchNorm2d(16),
        )
        # TODO   Change "22" according to the channel_num  of the DATASET
        self.depthwise = nn.Conv2d(16, 16, (3, 1), stride=1, padding=0, dilation=1, groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16 * 2, 1, 1, 0, 1, 1, bias=False)
        self.conv_block_2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Conv2d(32, 32, kernel_size=(1, 16), bias=False, padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 7)),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        out = self.conv_block_2(x)
        return out


class conv_block2(nn.Module):
    def __init__(self, ):
        super(conv_block2, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 32), bias=False, padding='same'),
            nn.BatchNorm2d(16),
        )
        # TODO   Change "22" according to the channel_num  of the DATASET
        self.depthwise = nn.Conv2d(16, 16, (3, 1), stride=1, padding=0, dilation=1, groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16 * 2, 1, 1, 0, 1, 1, bias=False)
        self.conv_block_2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Conv2d(32, 32, kernel_size=(1, 16), bias=False, padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 7)),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        out = self.conv_block_2(x)
        return out


class conv_block3(nn.Module):
    def __init__(self, ):
        super(conv_block3, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 16), bias=False, padding='same'),
            nn.BatchNorm2d(16),
        )
        # TODO   Change "22" according to the channel_num  of the DATASET
        self.depthwise = nn.Conv2d(16, 16, (3, 1), stride=1, padding=0, dilation=1, groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16 * 2, 1, 1, 0, 1, 1, bias=False)
        self.conv_block_2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Conv2d(32, 32, kernel_size=(1, 16), bias=False, padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 7)),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        out = self.conv_block_2(x)
        return out


class conv_block4(nn.Module):
    def __init__(self, ):
        super(conv_block4, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 8), bias=False, padding='same'),
            nn.BatchNorm2d(16),
        )
        # TODO   Change "22" according to the channel_num  of the DATASET
        self.depthwise = nn.Conv2d(16, 16, (3, 1), stride=1, padding=0, dilation=1, groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16 * 2, 1, 1, 0, 1, 1, bias=False)
        self.conv_block_2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Conv2d(32, 32, kernel_size=(1, 16), bias=False, padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 7)),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        out = self.conv_block_2(x)
        return out


class CSANet(nn.Module):
    def __init__(self, classes=4):
        super(CSANet, self).__init__()
        # TODO   Change "classes" according to the category of the DATASET in config!
        self.conv_block1 = conv_block1()
        self.conv_block2 = conv_block2()
        self.conv_block3 = conv_block3()
        self.conv_block4 = conv_block4()

        self.TCN_list = nn.ModuleList()
        for i in range(4):
            self.TCN_list.append(TCN_block())

        self.classify = nn.Linear(128, classes)
        self.msc1 = MSC(dim=32, kernel=[3, 5, 7], pad=[1, 2, 3], k1=2, k2=3)
        self.msc2 = MSC(dim=32, kernel=[3, 5, 7], pad=[1, 2, 3], k1=2, k2=3)
        self.msc3 = MSC(dim=32, kernel=[3, 5, 7], pad=[1, 2, 3], k1=2, k2=3)
        self.msc_nok = MSC_nok(dim=32, kernel=[3, 5, 7], pad=[1, 2, 3])

    def forward(self, x):
        # TODO Processed data dim should be 3, unsqueeze(1) has done in basemodel
        x1 = self.conv_block1(x)  # (B, 32, 1, T)
        x1 = x1.squeeze(2)  # (B, 32, T)
        out1 = self.msc_nok(x1, x1)  # (B, 32, T)
        x11 = out1 + x1
        x11 = self.TCN_list[0](x11)  # (B, 32, T)
        x11 = x11.mean(dim=-1)  # (B, 32)

        x2 = self.conv_block2(x)  # (B, 32, 1, T)
        x2 = x2.squeeze(2)  # (B, 32, T)
        out2 = self.msc1(x1, x2)  # (B, 32, T)
        x2 = out2 + x2
        x2 = self.TCN_list[1](x2)  # (B, 32, T)
        x2 = x2.mean(dim=-1)  # (B, 32)

        x3 = self.conv_block3(x)  # (B, 32, 1, T)
        x3 = x3.squeeze(2)  # (B, 32, T)
        out3 = self.msc2(x1, x3)  # (B, 32, T)
        x3 = out3 + x3
        x3 = self.TCN_list[2](x3)  # (B, 32, T)
        x3 = x3.mean(dim=-1)  # (B, 32)

        x4 = self.conv_block4(x)  # (B, 32, 1, T)
        x4 = x4.squeeze(2)  # (B, 32, T)
        out4 = self.msc3(x1, x4)  # (B, 32, T)
        x4 = out4 + x4
        x4 = self.TCN_list[3](x4)  # (B, 32, T)
        x4 = x4.mean(dim=-1)  # (B, 32)

        x = torch.cat((x11, x2, x3, x4), dim=1)  # (B, 128)
        x = self.classify(x)

        return x
