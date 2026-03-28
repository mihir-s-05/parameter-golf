"""
Segment-Bottleneck VAE Language Model for Parameter Golf

The sequence is split into fixed-size segments. Cross-segment information flows
ONLY through continuous latent vectors z. The VAE is the backbone:

  Encoder (bidirectional): compresses each S-token segment → z ~ N(μ, σ²)
  Prior (causal over z):   predicts p(z_i | z_{<i}) autoregressively
  Decoder (causal, local): predicts tokens given z + local causal context

The decoder CANNOT attend across segment boundaries. The latent z is the sole
mechanism for long-range information flow.

BPB = (reconstruction_nats + KL_nats) / total_bytes / ln(2)
"""

from __future__ import annotations

import copy
import glob
import io
import lzma
import math
import os
import random
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 2000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 200))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 32768))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 2700.0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    model_dim = int(os.environ.get("MODEL_DIM", 512))

    # Decoder
    dec_layers = int(os.environ.get("DEC_LAYERS", 6))
    dec_loops = int(os.environ.get("DEC_LOOPS", 1))
    dec_heads = int(os.environ.get("DEC_HEADS", 8))
    dec_kv_heads = int(os.environ.get("DEC_KV_HEADS", 4))
    dec_mlp_mult = int(os.environ.get("DEC_MLP_MULT", 3))

    # Encoder
    enc_dim = int(os.environ.get("ENC_DIM", 256))
    enc_layers = int(os.environ.get("ENC_LAYERS", 2))
    enc_heads = int(os.environ.get("ENC_HEADS", 4))

    # Prior
    prior_dim = int(os.environ.get("PRIOR_DIM", 256))
    prior_layers = int(os.environ.get("PRIOR_LAYERS", 3))
    prior_heads = int(os.environ.get("PRIOR_HEADS", 4))

    # VAE
    latent_dim = int(os.environ.get("LATENT_DIM", 128))
    segment_size = int(os.environ.get("SEGMENT_SIZE", 64))
    n_mem_tokens = int(os.environ.get("N_MEM_TOKENS", 4))
    kl_weight = float(os.environ.get("KL_WEIGHT", 1.0))
    free_bits = float(os.environ.get("FREE_BITS", 0.25))
    kl_warmup_steps = int(os.environ.get("KL_WARMUP_STEPS", 3000))

    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer
    embed_lr = float(os.environ.get("EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    latent_lr = float(os.environ.get("LATENT_LR", 0.01))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    # TTT (test-time training during eval)
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 0))
    use_prior_context = bool(int(os.environ.get("USE_PRIOR_CONTEXT", "1")))

    # QAT (quantization-aware training)
    qat_fraction = float(os.environ.get("QAT_FRACTION", "0.65"))
    latent_std_floor = float(os.environ.get("LATENT_STD_FLOOR", "1e-4"))
    latent_raw_std_clip = float(os.environ.get("LATENT_RAW_STD_CLIP", "8.0"))

CONTROL_NAMES = ("skip_weights", "attn_scale", "mlp_scale", "resid_mix", "q_gain",
                 "enc_pos", "prior_start", "lora", "loop_emb")
LATENT_NAMES = (
    "enc_down",
    "enc_pos",
    "mu_proj",
    "logvar_proj",
    "prior_in",
    "prior_start",
    "mu_prior_proj",
    "logvar_prior_proj",
    "z_to_mem",
)

# ---------------------------------------------------------------------------
# MUON OPTIMIZER
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            steps = group["backend_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if group["nesterov"] else buf.clone()
                g = zeropower_via_newtonschulz5(g, steps=steps)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g.to(p.dtype), alpha=-lr)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    assert header[0] == 20240520 and header[1] == 1, f"Bad header in {file}"
    num_tokens = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * 4)
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        assert self.files, f"No files for {pattern}"
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        parts: list[Tensor] = []
        rem = n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(rem, avail)
            parts.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            rem -= k
        return parts[0] if len(parts) == 1 else torch.cat(parts)


class TokenLoader:
    def __init__(self, pattern: str, device: torch.device):
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, total_tokens: int, seq_len: int) -> Tensor:
        n_seqs = total_tokens // seq_len
        raw = self.stream.take(n_seqs * seq_len).to(dtype=torch.int64)
        return raw.reshape(n_seqs, seq_len).to(self.device)


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    assert files, f"No files for {pattern}"
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = (tokens.numel() // seq_len) * seq_len
    return tokens[:usable]


def build_sentencepiece_luts(sp, vocab_size, device):
    sz = max(int(sp.vocab_size()), vocab_size)
    base_bytes = np.zeros(sz, dtype=np.int16)
    has_space = np.zeros(sz, dtype=np.bool_)
    is_boundary = np.ones(sz, dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(has_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary, dtype=torch.bool, device=device),
    )

# ---------------------------------------------------------------------------
# COMMON MODULES
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


def fake_quantize_ste(t: Tensor, clip_range: int = 31) -> Tensor:
    """Simulate int6 quantization with straight-through estimator."""
    t32 = t.float()
    if t32.ndim >= 2:
        amax = t32.abs().amax(dim=1, keepdim=True)
    else:
        amax = t32.abs().amax(keepdim=True)
    scale = (amax / clip_range).clamp_min(1.0 / clip_range)
    t_q = torch.clamp(torch.round(t32 / scale), -clip_range, clip_range)
    t_deq = (t_q * scale).to(t.dtype)
    return t + (t_deq - t).detach()


class CastedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self.qat and w.numel() > 65_536:
            w = fake_quantize_ste(w)
        return F.linear(x, w.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: Tensor | None = None
        self._sin: Tensor | None = None
        self._len = 0

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos is None or self._len < seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, None]
            self._sin = freqs.sin()[None, None]
            self._len = seq_len
        return self._cos[:, :, :seq_len].to(dtype), self._sin[:, :, :seq_len].to(dtype)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

# ---------------------------------------------------------------------------
# ENCODER (bidirectional, compresses segments → z)
# ---------------------------------------------------------------------------

class BidirAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.c_qkv = CastedLinear(dim, 3 * dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        qkv = self.c_qkv(x).reshape(B, T, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = BidirAttention(dim, n_heads)
        self.fc = CastedLinear(dim, dim * 2, bias=False)
        self.proj = CastedLinear(dim * 2, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.proj(F.leaky_relu(self.fc(self.mlp_norm(x)), 0.5).square())
        return x

# ---------------------------------------------------------------------------
# PRIOR (causal over z sequence, predicts p(z_i | z_{<i}))
# ---------------------------------------------------------------------------

class PriorAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope_base: float = 10000.0):
        super().__init__()
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.c_qkv = CastedLinear(dim, 3 * dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.rotary = Rotary(self.hd, rope_base)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        qkv = self.c_qkv(x).reshape(B, T, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        cos, sin = self.rotary(T, x.device, q.dtype)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class PriorBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope_base: float = 10000.0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = PriorAttention(dim, n_heads, rope_base)
        self.fc = CastedLinear(dim, dim * 2, bias=False)
        self.proj = CastedLinear(dim * 2, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.proj(F.leaky_relu(self.fc(self.mlp_norm(x)), 0.5).square())
        return x

# ---------------------------------------------------------------------------
# DECODER (causal within-segment, conditioned on z via memory tokens)
# ---------------------------------------------------------------------------

class DecoderAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 rope_base: float, qk_gain_init: float = 1.5, lora_rank: int = 0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv = n_kv_heads
        self.hd = dim // n_heads
        kv_dim = n_kv_heads * self.hd
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((n_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.hd, rope_base)
        self.lora_rank = lora_rank
        if lora_rank > 0:
            self.lora_q_A = nn.Parameter(torch.zeros(lora_rank, dim))
            self.lora_q_B = nn.Parameter(torch.zeros(dim, lora_rank))
            self.lora_v_A = nn.Parameter(torch.zeros(lora_rank, dim))
            self.lora_v_B = nn.Parameter(torch.zeros(kv_dim, lora_rank))

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        if self.lora_rank > 0:
            q = q + F.linear(F.linear(x, self.lora_q_A), self.lora_q_B)
            v = v + F.linear(F.linear(x, self.lora_v_A), self.lora_v_B)
        q = q.reshape(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = k.reshape(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = v.reshape(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = F.rms_norm(q, (self.hd,)), F.rms_norm(k, (self.hd,))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=(self.n_kv != self.n_heads))
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 mlp_mult: int, rope_base: float, qk_gain_init: float,
                 lora_rank: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = DecoderAttention(dim, n_heads, n_kv_heads, rope_base, qk_gain_init, lora_rank)
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.mlp_proj = CastedLinear(hidden, dim, bias=False)
        self.mlp_proj._zero_init = True
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None] * x + mix[1][None, None] * x0
        x = x + self.attn_scale.to(x.dtype)[None, None] * self.attn(self.attn_norm(x))
        h = F.leaky_relu(self.fc(self.mlp_norm(x)), 0.5).square()
        x = x + self.mlp_scale.to(x.dtype)[None, None] * self.mlp_proj(h)
        return x

# ---------------------------------------------------------------------------
# SEGMENT BOTTLENECK VAE
# ---------------------------------------------------------------------------

class SegmentBottleneckVAE(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        S = args.segment_size
        D = args.model_dim

        self.segment_size = S
        self.n_mem = args.n_mem_tokens
        self.vocab_size = args.vocab_size
        self.logit_softcap = args.logit_softcap
        self.latent_std_floor = args.latent_std_floor
        self.latent_raw_std_clip = args.latent_raw_std_clip

        # Shared token embedding
        self.tok_emb = nn.Embedding(args.vocab_size, D)
        nn.init.normal_(self.tok_emb.weight, std=0.005)

        # --- Encoder ---
        self.enc_down = CastedLinear(D, args.enc_dim, bias=False)
        self.enc_pos = nn.Parameter(torch.randn(S, args.enc_dim) * 0.02)
        self.enc_blocks = nn.ModuleList(
            [EncoderBlock(args.enc_dim, args.enc_heads) for _ in range(args.enc_layers)])
        self.enc_norm = RMSNorm()
        self.mu_proj = CastedLinear(args.enc_dim, args.latent_dim, bias=False)
        self.logvar_proj = CastedLinear(args.enc_dim, args.latent_dim, bias=False)
        nn.init.zeros_(self.logvar_proj.weight)

        # --- Prior ---
        self.prior_in = CastedLinear(args.latent_dim, args.prior_dim, bias=False)
        self.prior_start = nn.Parameter(torch.randn(args.prior_dim) * 0.02)
        self.prior_blocks = nn.ModuleList(
            [PriorBlock(args.prior_dim, args.prior_heads, args.rope_base)
             for _ in range(args.prior_layers)])
        self.prior_norm = RMSNorm()
        self.mu_prior_proj = CastedLinear(args.prior_dim, args.latent_dim, bias=False)
        self.logvar_prior_proj = CastedLinear(args.prior_dim, args.latent_dim, bias=False)
        nn.init.zeros_(self.logvar_prior_proj.weight)

        # --- Decoder ---
        self.use_prior_context = args.use_prior_context
        cond_dim = args.latent_dim + args.prior_dim if args.use_prior_context else args.latent_dim
        self.z_to_mem = CastedLinear(cond_dim, args.n_mem_tokens * D, bias=False)
        self.dec_loops = args.dec_loops
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(D, args.dec_heads, args.dec_kv_heads,
                         args.dec_mlp_mult, args.rope_base, args.qk_gain_init,
                         args.ttt_lora_rank)
            for _ in range(args.dec_layers)])
        if args.dec_loops > 1:
            self.loop_emb = nn.Parameter(torch.zeros(args.dec_loops, D))
        else:
            n_enc = args.dec_layers // 2
            n_dec = args.dec_layers - n_enc
            self.n_enc_layers = n_enc
            self.n_dec_layers = n_dec
            self.n_skip = min(n_enc, n_dec)
            self.skip_weights = nn.Parameter(torch.ones(self.n_skip, D, dtype=torch.float32))
        self.final_norm = RMSNorm()

        # Zero-init residual projections
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def _stable_logvar(self, raw: Tensor) -> Tensor:
        raw = raw.clamp(-self.latent_raw_std_clip, self.latent_raw_std_clip)
        std = F.softplus(raw) + self.latent_std_floor
        return 2.0 * torch.log(std)

    def encode(self, tok_embs: Tensor) -> tuple[Tensor, Tensor]:
        B, T, D = tok_embs.shape
        S = self.segment_size
        N = T // S
        x = tok_embs[:, :N * S].reshape(B * N, S, D)
        x = self.enc_down(x) + self.enc_pos[None]
        for blk in self.enc_blocks:
            x = blk(x)
        pooled = self.enc_norm(x).mean(dim=1)
        mu = self.mu_proj(pooled).reshape(B, N, -1)
        logvar = self._stable_logvar(self.logvar_proj(pooled).reshape(B, N, -1))
        return mu, logvar

    def prior_forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, N, _ = z.shape
        h = self.prior_in(z)
        start = self.prior_start.expand(B, 1, -1).to(h.dtype)
        h = torch.cat([start, h], dim=1)
        for blk in self.prior_blocks:
            h = blk(h)
        h = self.prior_norm(h[:, :N])
        mu_p = self.mu_prior_proj(h)
        logvar_p = self._stable_logvar(self.logvar_prior_proj(h))
        return mu_p, logvar_p, h

    def decode(self, cond: Tensor, tok_embs: Tensor, seg_tokens: Tensor) -> Tensor:
        B, T, D = tok_embs.shape
        S = self.segment_size
        N = T // S

        cond = F.rms_norm(cond, (cond.size(-1),))
        all_mems = self.z_to_mem(cond).reshape(B, N, self.n_mem, D)
        prev_mems = torch.cat([
            torch.zeros(B, 1, self.n_mem, D, device=all_mems.device, dtype=all_mems.dtype),
            all_mems[:, :-1]
        ], dim=1)
        mem = torch.cat([prev_mems, all_mems], dim=2).reshape(B * N, 2 * self.n_mem, D)

        seg_embs = tok_embs[:, :N * S].reshape(B, N, S, D)
        dec_tok = seg_embs[:, :, :-1].reshape(B * N, S - 1, D)
        x = torch.cat([mem, dec_tok], dim=1)
        x = F.rms_norm(x, (D,))
        x0 = x

        if self.dec_loops > 1:
            for loop_idx in range(self.dec_loops):
                x = x + self.loop_emb[loop_idx].to(x.dtype)[None, None]
                for blk in self.dec_blocks:
                    x = blk(x, x0)
        else:
            skips: list[Tensor] = []
            for i in range(self.n_enc_layers):
                x = self.dec_blocks[i](x, x0)
                skips.append(x)
            for i in range(self.n_dec_layers):
                if skips:
                    x = x + self.skip_weights[i].to(x.dtype)[None, None] * skips.pop()
                x = self.dec_blocks[self.n_enc_layers + i](x, x0)

        total_mem = 2 * self.n_mem
        h = self.final_norm(x)[:, total_mem - 1 :, :]
        logits = F.linear(h, self.tok_emb.weight)
        if self.logit_softcap > 0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        targets = seg_tokens.reshape(B * N, S)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size),
                               targets.reshape(-1), reduction="mean")

    def forward(self, tokens: Tensor, free_bits: float = 0.0) -> tuple[Tensor, Tensor, Tensor]:
        B, T = tokens.shape
        S = self.segment_size
        N = T // S

        tok_embs = self.tok_emb(tokens)
        seg_tokens = tokens[:, :N * S].reshape(B, N, S)

        mu, logvar = self.encode(tok_embs)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        mu_p, logvar_p, prior_ctx = self.prior_forward(z.detach())
        if self.use_prior_context:
            dec_cond = torch.cat([z, prior_ctx], dim=-1)
        else:
            dec_cond = z
        recon_loss = self.decode(dec_cond, tok_embs, seg_tokens)

        var_q = logvar.exp()
        var_p = logvar_p.exp().clamp_min(1e-8)
        kl_per_dim = 0.5 * (logvar_p - logvar + (var_q + (mu - mu_p).pow(2)) / var_p - 1)
        kl_true_loss = kl_per_dim.sum(dim=-1).mean() / self.segment_size
        kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)
        kl_loss = kl_per_dim.sum(dim=-1).mean() / self.segment_size

        return recon_loss, kl_loss, kl_true_loss

# ---------------------------------------------------------------------------
# QUANTIZATION (int6 GPTQ-lite + lzma)
# ---------------------------------------------------------------------------

def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    """GPTQ-lite: try multiple clip percentiles, pick lowest MSE."""
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float("inf")
        for pct in [0.999, 0.9995, 0.9999, 0.99999, 1.0]:
            row_clip = (torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0
                        else t32.abs().amax(dim=1))
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]),
                            -clip_range, clip_range).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()),
                    -clip_range, clip_range).to(torch.int8)
    return q, scale


def quantize_state_dict_int6(sd: dict[str, Tensor]):
    """Int6 [-31,31] for large tensors, fp16/fp32 passthrough for small ones."""
    result: dict[str, Tensor] = {}
    meta: dict[str, str] = {}
    for name, t in sd.items():
        t = t.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65_536:
            if any(cn in name for cn in CONTROL_NAMES):
                result[name] = t.float().contiguous()
            else:
                result[name] = t.half() if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(cn in name for cn in CONTROL_NAMES):
            result[name] = t.float().contiguous()
            meta[name] = "passthrough"
            continue
        q, s = quantize_int6_per_row(t)
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = "int6"
    return {"w": result, "m": meta}


def dequantize_state_dict_int6(obj, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    result, meta = obj["w"], obj["m"]
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        if info == "passthrough":
            t = result[name]
            if t.dtype == torch.float16 and orig.dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig.dtype)
            out[name] = t
        else:
            q, s = result[name + ".q"], result[name + ".scale"]
            if s.ndim > 0:
                out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig.dtype)
            else:
                out[name] = (q.float() * float(s.item())).to(orig.dtype)
    return out


def compress_model(sd: dict[str, Tensor]) -> bytes:
    quant_obj = quantize_state_dict_int6(sd)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    return lzma.compress(buf.getvalue(), preset=6)


def decompress_model(blob: bytes, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    obj = torch.load(io.BytesIO(lzma.decompress(blob)), map_location="cpu", weights_only=False)
    return dequantize_state_dict_int6(obj, template_sd)

# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()
                       if p.requires_grad}

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].lerp_(p.data, 1 - self.decay)

    def apply_to(self, model: nn.Module):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters()
                       if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])

# ---------------------------------------------------------------------------
# TTT (test-time training during evaluation)
# ---------------------------------------------------------------------------

def eval_val_ttt(
    args: Hyperparameters, raw_model: nn.Module, model: nn.Module,
    device: torch.device, val_tokens: Tensor,
    base_bytes_lut: Tensor, has_space_lut: Tensor, is_boundary_lut: Tensor,
    log,
) -> tuple[float, float, float]:
    """Legal score-first TTT: score each chunk, then train on already-scored data."""
    S = args.segment_size
    seq_len = args.train_seq_len
    total_seqs = val_tokens.numel() // seq_len
    batch_seqs = max(1, 65_536 // seq_len)
    total_seqs = (total_seqs // batch_seqs) * batch_seqs
    chunk_seqs = max(1, args.ttt_chunk_tokens // seq_len)
    num_chunks = (total_seqs + chunk_seqs - 1) // chunk_seqs

    has_lora = any("lora" in n for n, _ in raw_model.named_parameters())
    if has_lora:
        for n, p in raw_model.named_parameters():
            if "lora" in n:
                if "_A" in n:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                else:
                    nn.init.zeros_(p)
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        ttt_params = [p for n, p in raw_model.named_parameters() if "lora" in n]
        optimizer = torch.optim.Adam(ttt_params, lr=args.ttt_lr)
    else:
        ttt_params = [p for p in raw_model.parameters()]
        for p in ttt_params:
            p.requires_grad_(True)
        optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)

    total_recon = 0.0
    total_kl = 0.0
    total_tokens = 0
    total_bytes_d = 0.0
    t0 = time.perf_counter()

    log(f"TTT: {num_chunks} chunks, {chunk_seqs} seqs/chunk, "
        f"lr={args.ttt_lr}, epochs={args.ttt_epochs}")

    for ci in range(num_chunks):
        si = ci * chunk_seqs
        ei = min(si + chunk_seqs, total_seqs)

        # Phase 1: SCORE (no grads, no weight changes)
        model.eval()
        with torch.inference_mode():
            for bs in range(si, ei, batch_seqs):
                be = min(bs + batch_seqs, ei)
                chunk = val_tokens[bs * seq_len : be * seq_len]
                chunk = chunk.to(device=device, dtype=torch.int64).reshape(be - bs, seq_len)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    recon, kl, _kl_true = model(chunk, free_bits=0.0)
                n_tok = chunk.numel()
                total_recon += recon.item() * n_tok
                total_kl += kl.item() * n_tok
                total_tokens += n_tok
                flat = chunk.reshape(-1)
                prev = torch.cat([
                    torch.zeros(be - bs, 1, dtype=torch.int64, device=device),
                    chunk[:, :-1]], dim=1).reshape(-1)
                tb = base_bytes_lut[flat].to(torch.float64)
                tb += (has_space_lut[flat] & ~is_boundary_lut[prev]).to(torch.float64)
                total_bytes_d += tb.sum().item()

        # Phase 2: TRAIN on scored chunk (skip last)
        if ci < num_chunks - 1 and args.ttt_epochs > 0:
            model.train()
            cos_lr = args.ttt_lr * 0.5 * (1 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
            for pg in optimizer.param_groups:
                pg["lr"] = cos_lr
            for _ep in range(args.ttt_epochs):
                for bs in range(si, ei, batch_seqs):
                    be = min(bs + batch_seqs, ei)
                    chunk = val_tokens[bs * seq_len : be * seq_len]
                    chunk = chunk.to(device=device, dtype=torch.int64).reshape(be - bs, seq_len)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        recon, kl, _kl_true = model(chunk, free_bits=0.0)
                    (recon + kl).backward()
                    torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                    optimizer.step()

        if ci % 20 == 0 or ci == num_chunks - 1:
            bpb = (total_recon + total_kl) / total_bytes_d / math.log(2.0) if total_bytes_d > 0 else 0
            log(f"  ttt [{ci+1}/{num_chunks}] bpb={bpb:.4f} "
                f"time={time.perf_counter() - t0:.1f}s")

    bpb = (total_recon + total_kl) / total_bytes_d / math.log(2.0)
    recon_avg = total_recon / total_tokens
    kl_avg = total_kl / total_tokens
    model.eval()
    return recon_avg, bpb, kl_avg

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    args = Hyperparameters()
    code = Path(__file__).read_text(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt"
    def log(msg, console=True):
        if console:
            print(msg)
        with open(logfile, "a") as f:
            print(msg, file=f)

    assert args.train_seq_len % args.segment_size == 0, \
        f"seq_len ({args.train_seq_len}) must be divisible by segment_size ({args.segment_size})"

    n_segs = args.train_seq_len // args.segment_size
    log("Segment-Bottleneck VAE LM")
    log(f"  encoder: {args.enc_layers}L dim={args.enc_dim} heads={args.enc_heads}")
    log(f"  prior:   {args.prior_layers}L dim={args.prior_dim} heads={args.prior_heads}")
    log(f"  decoder: {args.dec_layers}L×{args.dec_loops} dim={args.model_dim} "
        f"heads={args.dec_heads}/{args.dec_kv_heads} mlp={args.dec_mlp_mult}x"
        f" lora_r={args.ttt_lora_rank}")
    log(f"  vae: seg={args.segment_size} latent={args.latent_dim} "
        f"mem={args.n_mem_tokens} segs/seq={n_segs}")
    log(f"  kl: weight={args.kl_weight} free_bits={args.free_bits} "
        f"warmup={args.kl_warmup_steps}")
    log(f"  opt: embed_lr={args.embed_lr} matrix_lr={args.matrix_lr} "
        f"latent_lr={args.latent_lr} scalar_lr={args.scalar_lr}")

    # Data
    train_loader = TokenLoader(args.train_files, device)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    log(f"  val tokens: {val_tokens.numel():,}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_space_lut, is_boundary_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device)

    # Model
    raw_model = SegmentBottleneckVAE(args).to(device).bfloat16()
    for m in raw_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    with torch.no_grad():
        for name, p in raw_model.named_parameters():
            if p.dtype != torch.float32:
                if p.ndim < 2 or any(cn in name for cn in CONTROL_NAMES):
                    p.data = p.data.float()

    n_params = sum(p.numel() for p in raw_model.parameters())
    log(f"  params: {n_params:,}")

    try:
        import torch._inductor.config as ind_cfg
        ind_cfg.triton.persistent_reductions = False
    except Exception:
        pass

    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    model = torch.compile(raw_model, dynamic=False)

    # Optimizer split: Muon for 2D matrix params, Adam for rest
    matrix_params: list[Tensor] = []
    latent_params: list[Tensor] = []
    scalar_params: list[Tensor] = []
    embed_params = [raw_model.tok_emb.weight]

    for name, p in raw_model.named_parameters():
        if p is raw_model.tok_emb.weight:
            continue
        if "lora" in name:
            continue
        if any(ln in name for ln in LATENT_NAMES):
            latent_params.append(p)
            continue
        if p.ndim == 2 and p.numel() > 256 and not any(cn in name for cn in CONTROL_NAMES):
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    optimizer_tok = torch.optim.Adam(
        [{"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2), fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr,
                          momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps)
    for g in optimizer_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    optimizer_latent = torch.optim.Adam(
        [{"params": latent_params, "lr": args.latent_lr, "base_lr": args.latent_lr}],
        betas=(args.beta1, args.beta2), fused=True)
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_latent, optimizer_scalar]

    ema = EMA(raw_model, args.ema_decay)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        wd_start = args.iterations - args.warmdown_iters
        if step >= wd_start:
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
        return 1.0

    def kl_weight_at(step: int) -> float:
        if step < args.kl_warmup_steps:
            return args.kl_weight * step / max(args.kl_warmup_steps, 1)
        return args.kl_weight

    # Validation
    @torch.no_grad()
    def validate(use_ema: bool = True) -> tuple[float, float, float]:
        if use_ema:
            ema.apply_to(raw_model)
        model.eval()

        S = args.segment_size
        total_seqs = val_tokens.numel() // args.train_seq_len
        batch_seqs = max(1, 65_536 // args.train_seq_len)
        total_seqs = (total_seqs // batch_seqs) * batch_seqs

        total_recon = 0.0
        total_kl = 0.0
        total_tokens = 0
        total_bytes_d = 0.0

        for si in range(0, total_seqs, batch_seqs):
            ei = min(si + batch_seqs, total_seqs)
            chunk = val_tokens[si * args.train_seq_len : ei * args.train_seq_len]
            chunk = chunk.to(device=device, dtype=torch.int64).reshape(ei - si, args.train_seq_len)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                recon, kl, _kl_true = model(chunk, free_bits=0.0)

            n_tok = chunk.numel()
            total_recon += recon.item() * n_tok
            total_kl += kl.item() * n_tok
            total_tokens += n_tok

            flat = chunk.reshape(-1)
            prev = torch.cat([
                torch.zeros(ei - si, 1, dtype=torch.int64, device=device),
                chunk[:, :-1]], dim=1).reshape(-1)
            tok_bytes = base_bytes_lut[flat].to(torch.float64)
            tok_bytes += (has_space_lut[flat] & ~is_boundary_lut[prev]).to(torch.float64)
            total_bytes_d += tok_bytes.sum().item()

        recon_per_tok = total_recon / total_tokens
        bpb = (total_recon + total_kl) / total_bytes_d / math.log(2.0)
        avg_kl = total_kl / total_tokens

        model.train()
        if use_ema:
            ema.restore(raw_model)
        return recon_per_tok, bpb, avg_kl

    # Training loop
    log("Training...")
    t0 = time.perf_counter()
    qat_start_step = int(args.qat_fraction * args.iterations)
    qat_active = False

    for step in range(args.iterations):
        elapsed = time.perf_counter() - t0
        if args.max_wallclock_seconds > 0 and elapsed > args.max_wallclock_seconds:
            log(f"Wallclock limit at step {step}")
            break

        if not qat_active and step >= qat_start_step:
            qat_active = True
            for m in raw_model.modules():
                if isinstance(m, CastedLinear):
                    m.qat = True
            ema = EMA(raw_model, args.ema_decay)
            log(f"QAT activated at step {step}, EMA reset")

        scale = lr_scale(step)
        kl_w = kl_weight_at(step)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g.get("base_lr", g["lr"]) * scale

        tokens = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len)
        zero_grad_all()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            recon, kl, kl_true = model(tokens, free_bits=args.free_bits)
        total_loss = recon + kl_w * kl
        total_loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()

        ema.update(raw_model)

        if step % args.train_log_every == 0:
            log(f"step={step:5d} recon={recon.item():.4f} kl={kl.item():.4f} "
                f"true_kl={kl_true.item():.4f} total={total_loss.item():.4f} kl_w={kl_w:.4f} "
                f"lr_s={scale:.3f} time={elapsed:.1f}s")

        if args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0:
            v_recon, v_bpb, v_kl = validate()
            log(f"VAL step={step} recon={v_recon:.4f} bpb={v_bpb:.4f} kl={v_kl:.4f}")

    # Final validation
    log("=" * 60)
    v_recon, v_bpb, v_kl = validate()
    log(f"Final: recon={v_recon:.4f} bpb={v_bpb:.4f} kl={v_kl:.4f}")

    # Save with EMA weights
    ema.apply_to(raw_model)

    compressed = compress_model(raw_model.state_dict())
    with open("final_model_vae2.int6.ptz", "wb") as f:
        f.write(compressed)

    code_bytes = len(code.encode("utf-8"))
    model_bytes = len(compressed)
    total_size = code_bytes + model_bytes
    log(f"Code: {code_bytes:,} | Model: {model_bytes:,} | "
        f"Total: {total_size:,} (limit: 16,000,000)")

    # Quantized round-trip eval
    template_sd = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
    with open("final_model_vae2.int6.ptz", "rb") as f:
        blob = f.read()
    q_state = decompress_model(blob, template_sd)
    raw_model.load_state_dict(q_state, strict=True)
    v_recon_q, v_bpb_q, v_kl_q = validate(use_ema=False)
    log(f"Quantized int6: recon={v_recon_q:.4f} bpb={v_bpb_q:.4f} kl={v_kl_q:.4f}")

    # TTT evaluation (on quantized model)
    if args.ttt_enabled:
        log("=" * 60)
        log("Running TTT evaluation...")
        ttt_recon, ttt_bpb, ttt_kl = eval_val_ttt(
            args, raw_model, model, device, val_tokens,
            base_bytes_lut, has_space_lut, is_boundary_lut, log)
        log(f"TTT final: recon={ttt_recon:.4f} bpb={ttt_bpb:.4f} kl={ttt_kl:.4f}")

    log("Done!")


if __name__ == "__main__":
    main()
