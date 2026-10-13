#!/usr/bin/env python3
"""v3 ACT decoder — replicates lerobot's ACTDecoder/ACTDecoderLayer exactly (same param
names/shapes, so a tactile=False layer state_dicts identically to the installed lerobot
version), plus an optional dedicated tactile-only cross-attention sub-block:

  self_attn -> multihead_attn(full memory) [-> tactile_cross_attn(10 tac tokens)] -> FFN

The tactile branch is summed into the full-memory cross-attn output before the residual
add (cheap, no extra params, symmetric treatment of both attention sources -- concat+proj
would need an extra dim-halving layer for no clear benefit at 10 tokens). Only active when
a layer is built with tactile=True AND a `tac_memory` is passed to forward(); otherwise
behaves identically to lerobot's ACTDecoderLayer (see design_v3_stage_memory_tactile.md §2.2).

We reimplement rather than subclass lerobot's ACTDecoderLayer because n_decoder_layers /
the tactile sub-block are per-project deltas -- the installed `lerobot` package is never
touched.
"""
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(name):
    return {"relu": F.relu, "gelu": F.gelu, "glu": F.glu}[name]


class ACTDecoderLayerV3(nn.Module):
    def __init__(self, cfg, tactile=False):
        super().__init__()
        self.tactile = tactile
        self.self_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)
        self.multihead_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)
        if tactile:
            self.tactile_cross_attn = nn.MultiheadAttention(cfg.dim_model, cfg.n_heads, dropout=cfg.dropout)

        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)

        self.norm1 = nn.LayerNorm(cfg.dim_model)
        self.norm2 = nn.LayerNorm(cfg.dim_model)
        self.norm3 = nn.LayerNorm(cfg.dim_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)

        self.activation = get_activation_fn(cfg.feedforward_activation)
        self.pre_norm = cfg.pre_norm      # our project always uses pre_norm=False (see _cfg)

    def _pe(self, t, pe):
        return t if pe is None else t + pe

    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None,
                tac_memory=None, tac_pos_embed=None):
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self._pe(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x; x = self.norm2(x)
        else:
            x = self.norm1(x); skip = x

        mh = self.multihead_attn(query=self._pe(x, decoder_pos_embed),
                                  key=self._pe(encoder_out, encoder_pos_embed),
                                  value=encoder_out)[0]
        if self.tactile and tac_memory is not None:
            tac = self.tactile_cross_attn(query=self._pe(x, decoder_pos_embed),
                                           key=self._pe(tac_memory, tac_pos_embed),
                                           value=tac_memory)[0]
            mh = mh + tac
        x = skip + self.dropout2(mh)
        if self.pre_norm:
            skip = x; x = self.norm3(x)
        else:
            x = self.norm2(x); skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


class ACTDecoderV3(nn.Module):
    def __init__(self, cfg, n_layers, tactile=False):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayerV3(cfg, tactile=tactile) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(cfg.dim_model)

    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None,
                tac_memory=None, tac_pos_embed=None):
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed,
                       tac_memory=tac_memory, tac_pos_embed=tac_pos_embed)
        return self.norm(x)
