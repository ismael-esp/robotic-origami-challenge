#!/usr/bin/env python3
"""FoldingACT — ACT-CVAE policy for origami folding.

Reuses lerobot's transformer blocks (ACTEncoder) but assembles OUR tokens:
  encoder input = [latent, proprio(153), fold_id, (aux x2), 4x image feature maps, (tactile x10)]
  CVAE encoder over [cls, proprio, action_chunk] -> latent (BERT-style, like ACT)
  decoder: `chunk` DETR queries cross-attend the encoder memory -> action chunk (K,65)

Per-role vision: shared ResNet-18 for the 2 head views, shared ResNet-18 for the 2 wrist
views (+ per-view id). Encoder is a config swap to DINOv3 ViT-S later.
v6: `n_cams=3` trains natively on the deployment contract's [head_left, wrist_left,
wrist_right] (no head_right slot: 3 views x n_img_tok image tokens instead of 4);
default n_cams=4 is byte-identical to v1-v5 and the state_dict keys/shapes are
identical in both modes (see __init__).

v3 delta over the trained v2 checkpoint (docs/research/design_v3_stage_memory_tactile.md),
all OFF by default so `FoldingACT(encoder=..., tactile=...)` still reproduces the exact v2
architecture/state_dict (`outputs/ckpt/best_v2_tactile.pt` loads unchanged):
  - aux_heads: completion/progress readout tokens + heads (§2.1)
  - n_decoder_layers: default bumped 1->3 (the LeRobot default of 1 replicates an upstream
    ACT bug, see §2.2/§2.3 configuration_act.py comment) -- pass 1 to reproduce v2 exactly
  - decoder_tactile_attn: dedicated tactile-only cross-attn sub-layer in the decoder (§2.2 item 2)
  - pre_encoder_tactile_attn: wrist-image<->tactile cross-attn before self.encoder (§2.2 item 1)
  - contact_gate: FoAR-style per-finger gate on the Sparsh tokens (§2.2 gate placement)

Smoke test: python policy/model/folding_act.py
"""
import os
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

from lerobot.policies.act.modeling_act import ACTEncoder, create_sinusoidal_pos_embedding

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparsh_encoder import SparshTactileEncoder            # noqa: E402
from act_decoder_v3 import ACTDecoderV3                    # noqa: E402

TORSO = list(range(58, 65))   # torso action dims -> down-weighted (anti-jitter)
DINO = {"dinov3": "vit_small_patch16_dinov3", "dinov2": "vit_small_patch14_dinov2"}

# Set FOLDING_ACT_SKIP_PRETRAINED=1 to build trunks with random init (no HF/torchvision
# downloads). ONLY valid when a full checkpoint state_dict is strict-loaded right after
# construction (e.g. policy/serve/policy_server.py) -- training must NOT set this.
SKIP_PRETRAINED = bool(os.environ.get("FOLDING_ACT_SKIP_PRETRAINED"))

# obs_lowdim = [state(65), joint_torque(58), tactile(30)] (fold_utils.Norm.obs_lowdim order).
# tactile(30) = 10 fingers x 3 kept force axes (verified against outputs/norm_stats.json
# keep_idx: [0,1,2, 6,7,8, ..., 54,55,56] -- first 3 of each 6-channel finger block).
N_TAC_FORCE_AXES = 3

# out: pred (B,K,65), mu/logvar (B,latent) as before; completion_logit/progress_pred are
# None unless aux_heads=True. A plain namedtuple keeps `pred, mu, lv = model(b)[:3]` working
# for call sites that don't need the aux outputs (eval_ckpt.py, build_training_report_v2.py),
# while train_full.py unpacks all 5 fields directly.
FoldingACTOutput = namedtuple("FoldingACTOutput", ["pred", "mu", "logvar", "completion_logit", "progress_pred"])


def _cfg(dim=512, heads=8, ff=1280, enc=4, dec=1, vae=4, drop=0.1):
    return SimpleNamespace(dim_model=dim, n_heads=heads, dim_feedforward=ff, dropout=drop,
                           feedforward_activation="relu", pre_norm=False,
                           n_encoder_layers=enc, n_decoder_layers=dec, n_vae_encoder_layers=vae)


class ResNetRole(nn.Module):
    """ResNet-18 trunk -> 49 tokens (7x7)."""
    n_tokens = 49
    def __init__(self, dim):
        super().__init__()
        m = torchvision.models.resnet18(
            weights=None if SKIP_PRETRAINED else torchvision.models.ResNet18_Weights.DEFAULT)
        self.body = nn.Sequential(*list(m.children())[:-2])   # -> (B,512,7,7)
        self.proj = nn.Conv2d(512, dim, 1)

    def forward(self, x):                                     # (B,3,224,224)
        f = self.proj(self.body(x))                           # (B,dim,7,7)
        return f.flatten(2).permute(2, 0, 1)                  # (49,B,dim)


class DinoRole(nn.Module):
    """DINOv3/v2 ViT-S -> patch tokens (fine-tunable)."""
    def __init__(self, dim, variant="dinov3"):
        super().__init__()
        self.backbone = timm.create_model(DINO[variant], pretrained=not SKIP_PRETRAINED, num_classes=0)
        self.backbone.set_grad_checkpointing(True)   # recompute in backward -> big VRAM saving
        self.n_prefix = self.backbone.num_prefix_tokens
        patch = self.backbone.patch_embed.patch_size[0]
        self.n_tokens = (224 // patch) ** 2                   # 196 (v3/16) or 256 (v2/14)
        self.proj = nn.Linear(self.backbone.embed_dim, dim)

    def forward(self, x):                                     # (B,3,224,224)
        t = self.backbone.forward_features(x)[:, self.n_prefix:]   # (B,N,embed) patches
        return self.proj(t).permute(1, 0, 2)                  # (N,B,dim)


def _make_role(encoder, dim):
    if encoder == "resnet18":
        return ResNetRole(dim)
    return DinoRole(dim, encoder)


class FoldingACT(nn.Module):
    def __init__(self, obs_dim=153, action_dim=65, chunk=45, n_folds=6, dim=512,
                 latent_dim=32, encoder="resnet18", tactile=False, n_fingers=10,
                 tactile_blocks=2, tactile_img=112,
                 n_decoder_layers=3, aux_heads=False, detach_aux=True,
                 decoder_tactile_attn=False, pre_encoder_tactile_attn=False,
                 n_pre_xattn_layers=1, contact_gate=False, n_tempos=0, n_cams=4):
        super().__init__()
        c = _cfg(dim=dim, dec=n_decoder_layers)
        self.dim, self.chunk, self.latent_dim, self.action_dim = dim, chunk, latent_dim, action_dim
        self.obs_dim = obs_dim
        self.tactile = tactile
        self.aux_heads = aux_heads
        self.detach_aux = detach_aux
        self.decoder_tactile_attn = tactile and decoder_tactile_attn
        self.pre_encoder_tactile_attn = tactile and pre_encoder_tactile_attn
        self.contact_gate = tactile and contact_gate

        # vision (per-role: head-pair / wrist-pair) + view id + learned spatial pos
        # v6: n_cams=4 (default) = [head_left, head_right, wrist_left, wrist_right],
        # byte-identical v1-v5 behavior. n_cams=3 = the deployment contract's native
        # [head_left, wrist_left, wrist_right]: NO head_right slot, the head encoder
        # consumes 1 view. View-id rows keep their v1-v5 semantics (0=head_left,
        # 2=wrist_left, 3=wrist_right; row 1 sits unused in 3-cam mode), so the
        # state_dict keys/shapes are IDENTICAL in both modes -- the camera count is
        # recorded as `n_cams` in the checkpoint dict (policy_server rebuilds from it).
        assert n_cams in (3, 4), f"n_cams must be 3 or 4, got {n_cams}"
        self.n_cams = n_cams
        self.cam_views = [0, 1, 2, 3] if n_cams == 4 else [0, 2, 3]
        self.n_head_views = n_cams - 2                # 2 head views, or 1 in 3-cam mode
        self.enc_head = _make_role(encoder, dim)
        self.enc_wrist = _make_role(encoder, dim)
        self.n_img_tok = self.enc_head.n_tokens
        self.view_embed = nn.Embedding(4, dim)
        self.img_spatial_pos = nn.Parameter(torch.randn(self.n_img_tok, dim) * 0.02)

        # tactile branch (Sparsh, init + light fine-tune) -> 10 fingertip tokens
        if tactile:
            self.enc_tactile = SparshTactileEncoder(n_fingers=n_fingers, out_dim=dim,
                                                    trainable_blocks=tactile_blocks, img=tactile_img)
            self.n_tac_tok = n_fingers
            self.tac_pos = nn.Parameter(torch.randn(n_fingers, dim) * 0.02)
            if self.contact_gate:                                       # FoAR-style per-finger gate (§2.2)
                self.noncontact_emb = nn.Parameter(torch.zeros(n_fingers, dim))
                nn.init.normal_(self.noncontact_emb, std=0.02)
            if self.pre_encoder_tactile_attn:                           # wrist-img <-> tactile, pre-self-attn
                self.pre_xattn = nn.ModuleList(
                    [nn.MultiheadAttention(dim, 8, dropout=0.1) for _ in range(n_pre_xattn_layers)])
                self.pre_xattn_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_pre_xattn_layers)])

        # proprio + fold token
        self.proprio_proj = nn.Linear(obs_dim, dim)
        self.fold_embed = nn.Embedding(n_folds, dim)

        # v5: tempo-conditioning token (retime.py time-warped training copies). 0 = no token,
        # byte-identical v1-v4 state_dict. When n_tempos>0, batch["tempo_id"] (index into the
        # training tempo list) is REQUIRED on every forward -- the token is appended at the
        # END of the encoder sequence (not next to the fold token) so the aux readout slots
        # keep their fixed memory[3]/memory[4] indices, and so DDP static_graph=True always
        # executes the embedding (no data-dependent branching).
        self.n_tempos = n_tempos
        if n_tempos:
            self.tempo_embed = nn.Embedding(n_tempos, dim)
            self.tempo_pos = nn.Parameter(torch.randn(1, dim) * 0.02)

        # completion/progress readout tokens (Octo-style, non-persistent -- reset every forward pass)
        if aux_heads:
            self.aux_query = nn.Embedding(2, dim)               # [0]=completion, [1]=progress
            self.aux_pos = nn.Parameter(torch.randn(2, dim) * 0.02)
            self.completion_head = nn.Linear(dim, 1)
            self.progress_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))

        # CVAE encoder
        self.vae_encoder = ACTEncoder(c, is_vae_encoder=True)
        self.cls = nn.Embedding(1, dim)
        self.vae_proprio_proj = nn.Linear(obs_dim, dim)
        self.vae_action_proj = nn.Linear(action_dim, dim)
        self.latent_out = nn.Linear(dim, latent_dim * 2)
        self.register_buffer("vae_pos", create_sinusoidal_pos_embedding(2 + chunk, dim).unsqueeze(1))
        self.latent_in = nn.Linear(latent_dim, dim)

        # transformer enc/dec
        self.encoder = ACTEncoder(c)
        self.decoder = ACTDecoderV3(c, n_decoder_layers, tactile=self.decoder_tactile_attn)
        self.tok_pos = nn.Parameter(torch.randn(3, dim) * 0.02)     # latent, proprio, fold
        self.dec_pos = nn.Parameter(torch.randn(chunk, dim) * 0.02)
        self.action_head = nn.Linear(dim, action_dim)

    def _img_tokens(self, images):
        """images (B,n_cams,3,224,224); slot order = cam_views order (4-cam: slot==view id,
        exact v1-v5 op sequence; 3-cam: slots [head_left, wrist_left, wrist_right])."""
        toks = []
        for slot, vid in enumerate(self.cam_views):
            enc = self.enc_head if vid < 2 else self.enc_wrist
            t = enc(images[:, slot])                    # (N,B,dim)
            t = t + self.view_embed.weight[vid][None, None]
            toks.append(t)
        img_tok = torch.cat(toks, 0)                    # (n_cams*N,B,dim)
        img_pos = self.img_spatial_pos.repeat(self.n_cams, 1)[:, None]  # (n_cams*N,1,dim)
        return img_tok, img_pos

    def _tactile_tokens(self, batch, obs, B):
        """Sparsh fingertip tokens, optionally FoAR-gated. (10,B,dim)."""
        tac = self.enc_tactile(batch["tactile_img"]).permute(1, 0, 2)   # (10,B,dim)
        if self.contact_gate:
            # UNCALIBRATED heuristic: L2 force magnitude per finger from the already-normalized
            # obs_lowdim tactile slice (last 30 dims = 10 fingers x 3 force axes), squashed to (0,1).
            force = obs[..., -(self.n_tac_tok * N_TAC_FORCE_AXES):].reshape(B, self.n_tac_tok, N_TAC_FORCE_AXES)
            phi = torch.sigmoid(3.0 * force.norm(dim=-1) - 1.5)          # (B,10)
            phi = phi.transpose(0, 1)[..., None]                        # (10,B,1)
            tac = phi * tac + (1 - phi) * self.noncontact_emb[:, None]
        return tac

    def forward(self, batch, use_vae=True):
        """`use_vae=True` (default, unchanged) reproduces the exact train-time/eval_ckpt.py
        behaviour: the CVAE encoder consumes `batch["action"]` (a real or teacher-forced
        future action chunk) to produce z. This is the ONLY path exercised by the saved
        checkpoints' state_dicts, so leaving the default untouched keeps every existing
        call site (train_full.py, eval_ckpt.py, rollout.py replay) byte-identical.

        `use_vae=False` is the true action-free deployment path flagged as missing in
        rollout.py's module docstring: no future action chunk exists on the real robot /
        Isaac Sim, so the CVAE encoder branch is skipped entirely and z is set to the CVAE
        prior's mean (zeros) -- standard ACT inference behaviour (ACT's `use_vae=False` at
        eval time; ELBO's KL term pushes q(z|x) toward N(0,1), so z=0 is the correct
        eval-mode substitute). `batch` need not contain "action" in this mode. mu/logvar
        are returned as None (no encoder ran, nothing to compute a KL loss against).
        """
        obs, fold = batch["obs_lowdim"], batch["fold_id"]
        B = obs.shape[0]

        if use_vae:
            # --- CVAE encoder (train): [cls, proprio, action_seq] -> latent ---
            action = batch["action"]
            cls = self.cls.weight[:, None].repeat(1, B, 1)          # (1,B,dim)
            pj = self.vae_proprio_proj(obs)[None]                   # (1,B,dim)
            aj = self.vae_action_proj(action).permute(1, 0, 2)      # (K,B,dim)
            vae_out = self.vae_encoder(torch.cat([cls, pj, aj], 0), pos_embed=self.vae_pos)
            mu, logvar = self.latent_out(vae_out[0]).chunk(2, -1)   # (B,latent) each
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            mu = logvar = None
            z = torch.zeros(B, self.latent_dim, device=obs.device, dtype=obs.dtype)

        # --- encoder tokens: [latent, proprio, fold, (aux), images, (tactile)] ---
        lat = self.latent_in(z)[None]                           # (1,B,dim)
        pro = self.proprio_proj(obs)[None]
        fol = self.fold_embed(fold)[None]
        img_tok, img_pos = self._img_tokens(batch["images"])

        tac = None
        if self.tactile:
            tac = self._tactile_tokens(batch, obs, B)                        # (10,B,dim), maybe gated
            if self.pre_encoder_tactile_attn:                                # wrist-img <-> tactile (§2.2 item 1)
                h = self.n_head_views * self.n_img_tok                       # head tokens precede the wrists
                wrist = img_tok[h:]                                          # (2n,B,dim)
                for attn, ln in zip(self.pre_xattn, self.pre_xattn_norm):
                    out, _ = attn(wrist, tac, tac)
                    wrist = ln(wrist + out)
                img_tok = torch.cat([img_tok[:h], wrist], 0)

        toks = [lat, pro, fol]
        poss = [self.tok_pos[:, None].expand(-1, B, -1)]
        if self.aux_heads:                                      # 2 non-persistent readout tokens (§2.1)
            aux = self.aux_query.weight[:, None].expand(-1, B, -1)
            toks.append(aux)
            poss.append(self.aux_pos[:, None].expand(-1, B, -1))
        toks.append(img_tok)
        poss.append(img_pos.expand(-1, B, -1))
        if self.tactile:
            toks.append(tac)
            poss.append(self.tac_pos[:, None].expand(-1, B, -1))
        if self.n_tempos:                                       # v5 tempo token (end of sequence, see __init__)
            toks.append(self.tempo_embed(batch["tempo_id"])[None])          # (1,B,dim)
            poss.append(self.tempo_pos[:, None].expand(-1, B, -1))
        enc_in = torch.cat(toks, 0)
        enc_pos = torch.cat(poss, 0)
        memory = self.encoder(enc_in, pos_embed=enc_pos)

        completion_logit = progress_pred = None
        if self.aux_heads:                                      # memory[3]/[4] = the 2 aux slots, fixed index
            c_in = memory[3].detach() if self.detach_aux else memory[3]
            p_in = memory[4].detach() if self.detach_aux else memory[4]
            completion_logit = self.completion_head(c_in)                    # (B,1)
            progress_pred = torch.sigmoid(self.progress_head(p_in))          # (B,1) in [0,1]

        # --- decoder: chunk queries -> action chunk ---
        q = torch.zeros(self.chunk, B, self.dim, device=obs.device)
        tac_memory = tac if self.decoder_tactile_attn else None
        tac_pe = self.tac_pos[:, None].expand(-1, B, -1) if tac_memory is not None else None
        dec_out = self.decoder(q, memory,
                               decoder_pos_embed=self.dec_pos[:, None].expand(-1, B, -1),
                               encoder_pos_embed=enc_pos, tac_memory=tac_memory, tac_pos_embed=tac_pe)
        pred = self.action_head(dec_out).permute(1, 0, 2)       # (B,K,action)
        return FoldingACTOutput(pred, mu, logvar, completion_logit, progress_pred)


def loss_fn(pred, mu, logvar, action, action_is_pad, kl_weight=10.0, torso_w=0.1,
            completion_logit=None, progress_pred=None, progress_target=None, aux_weight=0.1):
    w = torch.ones(action.shape[-1], device=pred.device)
    w[TORSO] = torso_w                                          # anti-jitter: damp torso
    l1 = (pred - action).abs() * w                             # (B,K,A)
    mask = (~action_is_pad).float()[..., None]                 # (B,K,1)
    l1 = (l1 * mask).sum() / (mask.sum() * action.shape[-1] + 1e-8)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
    total = l1 + kl_weight * kl
    parts = {"l1": float(l1.detach()), "kl": float(kl.detach())}

    if completion_logit is not None:                           # §2.1: wire the dead `progress` field
        completion_target = (progress_target > 0.9).float()    # "nearly done with this fold" ramp
        bce = F.binary_cross_entropy_with_logits(completion_logit.squeeze(-1), completion_target)
        prog = F.mse_loss(progress_pred.squeeze(-1), progress_target)
        total = total + aux_weight * bce + aux_weight * prog
        parts["bce"] = float(bce.detach()); parts["prog"] = float(prog.detach())

    return total, parts


def _smoke():
    import sys, time
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data"))
    from folding_dataset import FoldingDataset
    dev = "cuda:0"
    ds = FoldingDataset(split="train", seasons=["season_POC22061_2026_07_02_14_23_26_train"])
    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)))
    batch = {k: v.to(dev) for k, v in batch.items()}

    model = FoldingACT().to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    out = model(batch)
    loss, parts = loss_fn(out.pred, out.mu, out.logvar, batch["action"], batch["action_is_pad"])
    loss.backward()
    torch.cuda.synchronize()
    print(f"params: {nparam/1e6:.1f}M")
    print(f"tokens: encoder=199 (1 latent + 1 proprio + 1 fold + 196 image), decoder queries={model.chunk}")
    print(f"pred {tuple(out.pred.shape)}  loss={float(loss):.3f}  parts={parts}")
    print(f"fwd+bwd {time.time()-t0:.2f}s (cold)  peak VRAM {torch.cuda.max_memory_allocated(dev)/1e9:.2f} GB @ batch=4")
    print("OK ✅")


if __name__ == "__main__":
    _smoke()
