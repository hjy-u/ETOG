import torch
import torch.nn as nn
import torch.nn.functional as F
from model.clip import build_model
from .layers import Projector, FPN, FPN_vit
from .bridger import Bridger_SA_ViT, Bridger_SA_RN_fwd
from .decoder import TransformerDecoder

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1)) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return bce + dice

class ETOG_res(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        if "RN" in cfg.clip_pretrain:
            self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
            self.bridger = Bridger_SA_RN_fwd(d_model=cfg.ladder_dim, nhead=cfg.nhead, fusion_stage=cfg.multi_stage, word_dim=cfg.word_dim)
        else:
            self.backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.input_size).float()
            self.bridger = Bridger_SA_ViT(d_model=cfg.ladder_dim, nhead=cfg.nhead)
        # Fix Backbone
        for param_name, param in self.backbone.named_parameters():
            # if 'positional_embedding' not in param_name:
            param.requires_grad = False
        self.input_dim = 1024  # (clip visual input dim, after attnpooling)
        self.batchnorm = cfg.batchnorm
        self.lang_fusion_type = cfg.lang_fusion_type
        self.bilinear = cfg.bilinear
        self.up_factor = 2 if self.bilinear else 1
        self._build_decoder(cfg)
        self.loss = BCEDiceLoss()

    def _build_decoder(self, cfg):
        self.visual_sent_fpn = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, language_fuser=True, decoding=False)
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3) # cfg.vis_dim=512
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                          d_model=cfg.vis_dim,
                          nhead=cfg.num_head,
                          dim_ffn=cfg.dim_ffn,
                          dropout=cfg.dropout,
                          return_intermediate=cfg.intermediate)


    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        input_shape = img.shape[-2:]
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        im, word, state = self.bridger(img, word, self.backbone, pad_mask)
        x_sent = self.visual_sent_fpn(im, state)
        B, _, H, W = x_sent.size()
        vis, _ = self.decoder(x_sent, word, pad_mask=pad_mask)  # (x1 not need to LN, but x0 has to)
        vis = vis.reshape(B, -1, H, W)
        test_vis = vis.clone()
        x = self.proj(vis, state)
        x = F.interpolate(x, input_shape, mode='bilinear', align_corners=True)
        if self.training:
            loss = self.loss(x, mask)
            return x.detach(), mask, loss
        else:
            interm = torch.sum(test_vis, 1)
            interm = interm.unsqueeze(0)
            attention_vis = F.interpolate(interm, input_shape, mode='bilinear', align_corners=True)
            attention_vis = F.sigmoid(attention_vis)
            attention_vis = attention_vis.detach().cpu().numpy()[0, 0]
            return x.detach(), attention_vis

class ETOG_vit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        if "RN" in cfg.clip_pretrain:
            self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
            self.bridger = Bridger_SA_RN_fwd(d_model=cfg.ladder_dim, nhead=cfg.nhead, fusion_stage=cfg.multi_stage)
        else:
            self.backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.input_size).float()
            self.bridger = Bridger_SA_ViT(d_model=cfg.ladder_dim, nhead=cfg.nhead)
        # Fix Backbone
        for param_name, param in self.backbone.named_parameters():
            if 'positional_embedding' not in param_name:
                param.requires_grad = False
        self.input_dim = 1024  # (clip visual input dim, after attnpooling)
        self.batchnorm = cfg.batchnorm
        self.lang_fusion_type = cfg.lang_fusion_type
        self.bilinear = cfg.bilinear
        self.up_factor = 2 if self.bilinear else 1
        self._build_decoder(cfg)
        self.loss = BCEDiceLoss()

    def _build_decoder(self, cfg):
        self.visual_sent_fpn = FPN_vit(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, language_fuser=True, decoding=False)
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3) # cfg.vis_dim=512
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                          d_model=cfg.vis_dim,
                          nhead=cfg.num_head,
                          dim_ffn=cfg.dim_ffn,
                          dropout=cfg.dropout,
                          return_intermediate=cfg.intermediate)

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        input_shape = img.shape[-2:]
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        im, word, state = self.bridger(img, word, self.backbone, pad_mask)
        x_sent = self.visual_sent_fpn(im, state)
        B, _, H, W = x_sent.size()
        vis, _ = self.decoder(x_sent, word, pad_mask=pad_mask)  # (x1 not need to LN, but x0 has to)
        vis = vis.reshape(B, -1, H, W)
        x = self.proj(vis, state)
        x = F.interpolate(x, input_shape, mode='bilinear', align_corners=True)
        if self.training:
            loss = self.loss(x, mask)
            return x.detach(), mask, loss
        else:
            return x.detach(), None