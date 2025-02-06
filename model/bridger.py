'''
Code manily adapted from https://github.com/kkakkkka/ETRIS/blob/main/model/bridger.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import VLSAadapter

# add forward output, current best, no change; prenorm
class Bridger_SA_RN_fwd(nn.Module):
    def __init__(self,
                 d_img = [512, 1024, 2048],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = [2, 1, 2],
                 num_layers = 12,
                 fusion_stage = 3,
                 word_dim = 1024
                ):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.fusion_stage = fusion_stage

        self.fusion= nn.ModuleList()
        self.zoom_in = nn.ModuleList()
        self.zoom_out = nn.ModuleList()
        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.ln_v = nn.ModuleList()
        # self.v_gate = nn.ModuleList()
        # self.t_gate = nn.ModuleList()
        self.ln_t = nn.ModuleList()
        for i in range(num_stages):
            if i >= num_stages - fusion_stage:
                self.fusion.append(VLSAadapter(d_model=d_model, nhead=nhead))
                if i < num_stages - 1:
                    self.zoom_in.append(nn.Conv2d(d_img[i], d_model, kernel_size=strides[i], stride=strides[i], bias=False))
                    self.zoom_out.append(nn.ConvTranspose2d(d_model, d_img[i], kernel_size=strides[i], stride=strides[i], bias=False))
                    self.linear1.append(nn.Linear(d_txt, d_model))
                    self.linear2.append(nn.Linear(d_model, d_txt))
                    # if fusion_stage > 1:
                    self.ln_v.append(nn.LayerNorm(d_model))
                    self.ln_t.append(nn.LayerNorm(d_model))
                else:
                    self.zoom_in.append(nn.ConvTranspose2d(d_img[i], d_model ,kernel_size=strides[i], stride=strides[i], bias=False))
                    self.zoom_out.append(nn.Conv2d(d_model, d_img[i], kernel_size=strides[i], stride=strides[i], bias=False))
                    self.linear1.append(nn.Linear(d_txt, d_model))
                    self.linear2.append(nn.Linear(d_model, d_txt))
                    self.ln_v.append(nn.LayerNorm(d_model))
                    self.ln_t.append(nn.LayerNorm(d_model))
            else:
                self.fusion.append(None)
                # self.fusion_t.append(None)
                self.zoom_in.append(None)
                self.zoom_out.append(None)
                self.linear1.append(None)
                self.linear2.append(None)
                self.ln_v.append(None)
                self.ln_t.append(None)
        #change last_conv for res50 : outchannels 1024, res101: outchannels 512
        self.last_conv = nn.Conv2d(d_model, word_dim, kernel_size=strides[-1], stride=strides[-1], bias=False)
        self.last_linear = nn.Linear(d_model, d_txt)
        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                # m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, vis, text, backbone, pad_mask):
        def stem(x):
            for conv, bn in [(vis_enc.conv1, vis_enc.bn1), (vis_enc.conv2, vis_enc.bn2),
                             (vis_enc.conv3, vis_enc.bn3)]:
                x = vis_enc.relu(bn(conv(x)))
            x = vis_enc.avgpool(x)
            return x
        # vision
        vis_enc = backbone.visual
        vis = vis.type(vis_enc.conv1.weight.dtype)
        vis = stem(vis)
        vis = vis_enc.layer1(vis)
        vis_enc_layers = [vis_enc.layer2, vis_enc.layer3, vis_enc.layer4]

        # language
        txt = backbone.token_embedding(text).type(
            backbone.dtype)  # [batch_size, n_ctx, d_model]

        txt_enc = backbone.transformer
        txt = txt + backbone.positional_embedding.type(backbone.dtype)[:txt.size(1)]
        txt = txt.permute(1, 0, 2)  # NLD -> LND

        # fusion
        stage_i = 0
        vis_outs = []
        forward_out = []
        for i in range(self.num_layers):
            if (i+1)%4 != 0:
                txt = txt_enc.resblocks[i](txt)
            else:
                # feed into this layer
                txt = txt_enc.resblocks[i](txt)
                vis = vis_enc_layers[stage_i](vis)
                if stage_i >= self.num_stages - self.fusion_stage:
                    # residual operation
                    v = vis.clone()
                    t = txt.clone() #t : N_l, B, dim
                    # dimension reduction
                    v = self.zoom_in[stage_i](v)
                    t = self.linear1[stage_i](t)
                    # multi modal fusion
                    B, C, H, W = v.shape
                    v = v.reshape(B, C, -1).permute(2, 0, 1) # B, C, H, W -> B, C, HW -> HW, B, C(676, 64, 256)
                    if stage_i == 0:
                        v, t = self.ln_v[stage_i](v), self.ln_t[stage_i](t)
                    if stage_i > 0:
                        v, t = self.ln_v[stage_i](v+last_v), self.ln_t[stage_i](t+last_t)
                    v, t = self.fusion[stage_i](v, t)
                    last_v, last_t = v, t # LND
                    v = v.permute(1, 2, 0).reshape(B, -1, H, W) # HW, B, C -> B, C, HW -> B, C, H, W
                    if stage_i == 2:
                        forward_out.append(v)
                        forward_out.append(t)
                    # dimension recovery
                    v = self.zoom_out[stage_i](v)
                    t = self.linear2[stage_i](t)
                    # residual connect
                    vis = vis + v
                    txt = txt + t
                stage_i += 1
                if stage_i < self.num_stages:
                    vis_outs.append(vis)
        # After fusion
        vis = vis_enc.attnpool(vis)

        forward_vis = self.last_conv(forward_out[0])
        vis = vis + forward_vis
        forward_t = self.last_linear(forward_out[1])
        txt = txt + forward_t

        vis_outs.append(vis)

        # language
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        txt = backbone.ln_final(txt).type(backbone.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = txt[torch.arange(txt.shape[0]),
                  text.argmax(dim=-1)] @ backbone.text_projection

        # forward
        return vis_outs, txt, state

class Bridger_SA_ViT(nn.Module):
    def __init__(self,
                 d_img=[768, 768, 768],
                 d_txt=512,
                 d_model=64,
                 nhead=8,
                 num_stages=3,
                 strides=[2, 2, 2],
                 num_layers=12,
                 shared_weights=False,
                 ):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers

        self.zoom_in, self.zoom_out = nn.ModuleList(), nn.ModuleList()
        self.linear1, self.linear2 = nn.ModuleList(), nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.ln_v = nn.ModuleList()
        self.ln_t = nn.ModuleList()
        for i in range(num_stages):
            self.fusion.append(VLSAadapter(d_model=d_model, nhead=nhead))
            # self.fusion.append(VLSAadapter_1(d_model=d_model, nhead=nhead))
            self.linear1.append(nn.Linear(d_txt, d_model))
            self.linear2.append(nn.Linear(d_model, d_txt))
            self.zoom_in.append(nn.Linear(d_img[i], d_model))
            self.zoom_out.append(nn.Linear(d_model, d_img[i]))
            self.ln_v.append(nn.LayerNorm(d_model))
            self.ln_t.append(nn.LayerNorm(d_model))
        self.last_conv = nn.Linear(d_model, 512)
        self.last_linear = nn.Linear(d_model, d_txt)
        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                # m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, text, backbone, pad_mask=None):
        # vision
        img = img.type(backbone.dtype)
        vis_enc = backbone.visual
        vis = vis_enc.conv1(img)  # shape = [*, width, grid, grid]
        vis = vis.reshape(vis.shape[0], vis.shape[1], -1)  # shape = [*, width, grid ** 2]
        vis = vis.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        vis = torch.cat([
            vis_enc.class_embedding.to(vis.dtype) + torch.zeros(vis.shape[0], 1, vis.shape[-1],
                                                                dtype=vis.dtype, device=vis.device), vis],
            dim=1)  # shape = [*, grid ** 2 + 1, width]
        vis = vis + vis_enc.positional_embedding.to(vis.dtype)
        vis = vis_enc.ln_pre(vis)

        vis = vis.permute(1, 0, 2)  # NLD -> LND

        # language
        txt = backbone.token_embedding(text).type(
            backbone.dtype)  # [batch_size, n_ctx, d_model]

        txt_enc = backbone.transformer
        txt = txt + backbone.positional_embedding.type(backbone.dtype)[:txt.size(1)]
        txt = txt.permute(1, 0, 2)  # NLD -> LND

        # fusion
        stage_i = 0
        vis_outs = []
        forward_out = []
        for i in range(self.num_layers):
            if (i + 1) % 4 != 0:
                vis = vis_enc.transformer.resblocks[i](vis)
                txt = txt_enc.resblocks[i](txt)
            else:
                vis = vis_enc.transformer.resblocks[i](vis)
                txt = txt_enc.resblocks[i](txt)
                # residual operation
                v = vis.clone() # LND 197, 16, 768
                t = txt.clone()

                v = self.zoom_in[stage_i](v)
                t = self.linear1[stage_i](t)
                if stage_i == 0:
                    v, t = self.ln_v[stage_i](v), self.ln_t[stage_i](t)
                if stage_i > 0:
                    v, t = self.ln_v[stage_i](v + last_v), self.ln_t[stage_i](t + last_t)
                # multi modal fusion
                v, t = self.fusion[stage_i](v, t)
                last_v, last_t = v, t
                if stage_i == 2:
                    forward_out.append(v)  # LND (196, 16, 64)
                    forward_out.append(t)
                # dimension recovery
                v = self.zoom_out[stage_i](v)
                t = self.linear2[stage_i](t)
                # residual connect
                vis= vis + v
                txt = txt + t
                stage_i += 1
                if stage_i < self.num_stages:
                    vis_out = vis[1:, :, :].permute(1, 2, 0)  # N, D, L
                    B, C, L = vis_out.shape
                    H = int(L ** 0.5)
                    W = L // H
                    vis_out = vis_out.reshape(B, C, H, W)  # B, D, H, W
                    vis_out = F.interpolate(vis_out, scale_factor=2 // stage_i, mode='bilinear')
                    vis_outs.append(vis_out)

        # After fusion
        # vision
        # 197, 64, 768 -> 64, 197, 768
        vis = vis.permute(1, 0, 2)  # LND -> NLD

        # x = vis_enc.ln_post(x[:, 0, :])
        # 64, 197, 768 -> 64, 196, 768
        vis = vis_enc.ln_post(vis)

        if vis_enc.proj is not None:
            vis = vis @ vis_enc.proj

        # 64, 196, 512 -> 64, 512, 196
        B, N, C = vis[:,1:,:].shape
        H = int(N ** 0.5)
        W = N // H
        vis = vis[:,1:,:].permute(0, 2, 1).reshape(B, C, H, W)  # B, N, D -> B, D, N -> B, D, H, W

        forward_vis = self.last_conv(forward_out[0]) #LND 197, 16, 512
        forward_vis = forward_vis[1:,...].reshape(B, C, H, W)
        vis = vis + forward_vis
        forward_t = self.last_linear(forward_out[1])
        txt = txt + forward_t

        vis_outs.append(vis)

        # language
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        txt = backbone.ln_final(txt).type(backbone.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = txt[torch.arange(txt.shape[0]),
        text.argmax(dim=-1)] @ backbone.text_projection

        # forward
        output = vis_outs, txt, state

        return output