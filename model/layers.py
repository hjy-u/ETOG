import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import model.fusion as fusion
from einops import rearrange

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

def conv_layer_group(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.GroupNorm(32, out_dim), nn.ReLU(True))

def deconv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None, act='gelu'):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU() if act =='gelu' else nn.ReLU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 1024
        '''
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        # b, 1, 104, 104
        return out

# no padding mask;
class VLSAadapter(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.ffn = FeedForward(d_model, dim_feedforward, dropout=dropout, act='relu')
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        spatial_shape = [tgt.shape[0], memory.shape[0]]
        #cat text with visual
        encoder_input_list = [tgt, memory] # vis + txt
        encoder_input = torch.cat(encoder_input_list, dim=0)
        tgt2 = self.self_attn(query=encoder_input, key=encoder_input, value=encoder_input)[0]
        tgt = encoder_input + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        vistxt = torch.split(tgt, spatial_shape, dim=0)
        vis = vistxt[0]
        txt = vistxt[1]
        return vis, txt
        # self attn
        # q = k = self.with_pos_embed(tgt, query_pos)
        # v = tgt
        # tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
        #                       key_padding_mask=tgt_key_padding_mask)[0] # [H*W, B, C]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        #
        # # cross attn
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=None,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        #
        # # ffn
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        # return tgt

# do not modify
class FPN(nn.Module):
    def __init__(self, in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024], language_fuser=True, decoding=False):
        super(FPN, self).__init__()

        self.proj_input_dim = in_channels[-1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lang_fusion_type = 'mult'
        self.language_fuser = language_fuser

        self.conv0 = conv_layer(in_channels[2], out_channels[1], 1, 0)
        self.conv1 = conv_layer(in_channels[1], out_channels[1], 1, 0)
        self.conv2 = conv_layer(in_channels[0], out_channels[1], 1, 0)

        if language_fuser:
            self.lang_proj0 = nn.Linear(self.proj_input_dim, out_channels[1])
            self.lang_fuser0 = fusion.names[self.lang_fusion_type](input_dim=self.in_channels[1])
            self.lang_proj1 = nn.Linear(self.proj_input_dim, out_channels[1])
            self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.in_channels[1])
            self.lang_proj2 = nn.Linear(self.proj_input_dim, out_channels[1])
            self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.in_channels[1])

        self.convp4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.convp3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.convp2 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.coordconv = nn.Sequential(conv_layer(3*out_channels[1], out_channels[1], 3, 1))

        # self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, imgs, sent_emb=None):
        x2, x3, x4 = imgs
        p4 = self.conv0(x4)
        if self.language_fuser:
            p4 = self.lang_fuser0(p4, sent_emb, x2_mask=None, x2_proj=self.lang_proj0)
        p4_up = F.interpolate(p4, scale_factor=2, mode='bilinear')

        x3 = self.conv1(x3)
        p3 = x3 + p4_up
        if self.language_fuser:
            p3 = self.lang_fuser1(p3, sent_emb, x2_mask=None, x2_proj=self.lang_proj1)
        p3_up = F.interpolate(p3, scale_factor=2, mode='bilinear')

        x2 = self.conv2(x2)
        p2 = x2 + p3_up
        if self.language_fuser:
            p2 = self.lang_fuser2(p2, sent_emb, x2_mask=None, x2_proj=self.lang_proj2)

        f4 = self.convp4(p4)
        f4 = F.interpolate(f4, scale_factor=2, mode='bilinear')

        f3 = self.convp3(p3)
        f2 = self.convp2(p2)
        f2 = F.avg_pool2d(f2, 2, 2)

        fv = torch.cat([f4, f3, f2], dim=1)
        fv = self.coordconv(fv)

        return fv

class FPN_vit(nn.Module):
    def __init__(self, in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024], language_fuser=True, decoding=False):
        super(FPN_vit, self).__init__()

        self.proj_input_dim = in_channels[-1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lang_fusion_type = 'mult'
        self.language_fuser = language_fuser

        self.conv0 = conv_layer(in_channels[2], out_channels[1], 1, 0)
        self.conv1 = conv_layer(in_channels[1], out_channels[1], 1, 0)
        self.conv2 = conv_layer(in_channels[0], out_channels[1], 1, 0)

        if language_fuser:
            self.lang_proj0 = nn.Linear(self.proj_input_dim, out_channels[1])
            self.lang_fuser0 = fusion.names[self.lang_fusion_type](input_dim=self.in_channels[1])
            self.lang_proj1 = nn.Linear(self.proj_input_dim, out_channels[1])
            self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.in_channels[1])
            self.lang_proj2 = nn.Linear(self.proj_input_dim, out_channels[1])
            self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.in_channels[1])

        self.convp4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.convp3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.convp2 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.coordconv = nn.Sequential(conv_layer(3*out_channels[1], out_channels[1], 3, 1))

        # self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, imgs, sent_emb=None):
        x2, x3, x4 = imgs
        p4 = self.conv0(x4)
        if self.language_fuser:
            p4 = self.lang_fuser0(p4, sent_emb, x2_mask=None, x2_proj=self.lang_proj0)
        p4_up = p4#F.interpolate(p4, scale_factor=2, mode='bilinear')

        x3 = self.conv1(x3)
        p3 = x3 + p4_up
        if self.language_fuser:
            p3 = self.lang_fuser1(p3, sent_emb, x2_mask=None, x2_proj=self.lang_proj1)
        p3_up = F.interpolate(p3, scale_factor=2, mode='bilinear')

        x2 = self.conv2(x2)
        p2 = x2 + p3_up
        if self.language_fuser:
            p2 = self.lang_fuser2(p2, sent_emb, x2_mask=None, x2_proj=self.lang_proj2)

        f4 = self.convp4(p4)
        # f4 = F.interpolate(f4, scale_factor=2, mode='bilinear')

        f3 = self.convp3(p3)
        f2 = self.convp2(p2)
        f2 = F.avg_pool2d(f2, 2, 2)

        fv = torch.cat([f4, f3, f2], dim=1)
        fv = self.coordconv(fv)

        return fv