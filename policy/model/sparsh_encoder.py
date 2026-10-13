#!/usr/bin/env python3
"""Sparsh tactile encoder — DINOv3 ViT-B initialised from facebook/sparsh, then
LIGHTLY fine-tuned (patch_embed + last-N blocks + final norm trainable; the rest
frozen).  This is the robust path chosen over strict-frozen: Sparsh stores its RoPE
as `frequency_bands (2,192)` (full-dim 2D axial) which no public loader reproduces,
so instead of a fragile RoPE reimplementation we keep timm's DINOv3 RoPE and let a
little fine-tuning repair the positional-scheme mismatch.  All 12 blocks' learned
weights load faithfully (verified: missing=[]).

Input : (B, F, 6, H, W)  — F fingertips, 6ch = I_t ⊕ I_{t-5} (temporal pair)
Output: (B, F, out_dim)   — one pooled token per fingertip (finger-id added)

Sparsh weight remap (checkpoint -> timm vit_base_patch16_dinov3_qkvb):
  attn.qkv.bias (3D)  -> attn.q_bias + attn.v_bias  (drop k_bias == 0)
  ls1.gamma/ls2.gamma -> gamma_1/gamma_2            (layerscale naming)
  register_tokens     -> reg_token
  patch_embed.proj    -> loads directly (both 6-channel)
"""
import os
from functools import lru_cache

import timm
import torch
import torch.nn as nn

# See folding_act.py: skip the HF facebook/sparsh download when a full checkpoint
# will be strict-loaded over this module anyway (offline serving / bundle).
SKIP_PRETRAINED = bool(os.environ.get("FOLDING_ACT_SKIP_PRETRAINED"))


@lru_cache(maxsize=1)
def _sparsh_state():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    return load_file(hf_hub_download("facebook/sparsh", "dino_vitbase.safetensors"))


def _remap(sp):
    out = {}
    for k, v in sp.items():
        if k.endswith("attn.qkv.bias"):
            q, _k, vv = v.chunk(3)
            out[k.replace("qkv.bias", "q_bias")] = q
            out[k.replace("qkv.bias", "v_bias")] = vv
            continue
        out[(k.replace("ls1.gamma", "gamma_1").replace("ls2.gamma", "gamma_2")
              .replace("register_tokens", "reg_token"))] = v
    return out


class SparshTactileEncoder(nn.Module):
    def __init__(self, n_fingers=10, out_dim=512, trainable_blocks=2, img=112):
        super().__init__()
        self.n_fingers = n_fingers
        self.vit = timm.create_model(
            "vit_base_patch16_dinov3_qkvb", pretrained=False, num_classes=0,
            in_chans=6, num_reg_tokens=1, class_token=False,
            img_size=img, dynamic_img_size=True)
        if not SKIP_PRETRAINED:
            missing, unexpected = self.vit.load_state_dict(_remap(_sparsh_state()), strict=False)
            assert not missing, f"Sparsh load missing keys: {missing}"    # frequency_bands is regenerated
        self.vit.set_grad_checkpointing(True)    # recompute in backward: patch_embed is trainable
        self._freeze(trainable_blocks)

        d = self.vit.embed_dim
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, out_dim)
        self.finger_emb = nn.Parameter(torch.zeros(n_fingers, out_dim))
        nn.init.normal_(self.finger_emb, std=0.02)
        self.out_dim = out_dim

    def _freeze(self, trainable_blocks):
        for p in self.vit.parameters():
            p.requires_grad_(False)
        n = len(self.vit.blocks)
        keep = set(range(n - trainable_blocks, n))
        for name, p in self.vit.named_parameters():
            if name.startswith("patch_embed") or name.startswith("norm"):
                p.requires_grad_(True)
            if name.startswith("blocks."):
                if int(name.split(".")[1]) in keep:
                    p.requires_grad_(True)

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):                                # x: (B, F, 6, H, W)
        B, F = x.shape[:2]
        assert F == self.n_fingers, f"expected {self.n_fingers} fingers, got {F}"
        feats = self.vit.forward_features(x.flatten(0, 1))       # (B*F, 1+P, d)
        patch = feats[:, self.vit.num_prefix_tokens:]           # drop register token
        pooled = self.norm(patch.mean(1))                       # (B*F, d) mean-pool patches
        tok = self.proj(pooled).view(B, F, self.out_dim)        # (B, F, out_dim)
        return tok + self.finger_emb[None]                      # + finger-id


if __name__ == "__main__":
    enc = SparshTactileEncoder(n_fingers=10, out_dim=512, trainable_blocks=2, img=112)
    tot = sum(p.numel() for p in enc.parameters())
    print(f"params: {tot/1e6:.1f}M total | {enc.n_trainable()/1e6:.1f}M trainable "
          f"({100*enc.n_trainable()/tot:.0f}%)")
    x = torch.randn(4, 10, 6, 112, 112)
    y = enc(x)
    print("forward:", tuple(x.shape), "->", tuple(y.shape))
    y.sum().backward()
    g = [n for n, p in enc.named_parameters() if p.requires_grad and p.grad is not None]
    print(f"backward OK; {len(g)} trainable tensors got grads")
    print("SPARSH ENCODER READY ✅")
