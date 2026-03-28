#!/bin/bash
set -e

echo "=============================================="
echo "Segment-Bottleneck VAE LM — Single H100 Stability Probe"
echo "=============================================="

pip install -q sentencepiece 2>/dev/null || true

export RUN_ID="sbvae_h100_probe_$(date +%Y%m%d_%H%M%S)"
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export SEED=1337
export VOCAB_SIZE=1024

# Decoder (7 unique layers, no weight tying)
export MODEL_DIM=512
export DEC_LAYERS=7
export DEC_LOOPS=1
export DEC_HEADS=8
export DEC_KV_HEADS=4
export DEC_MLP_MULT=3
export LOGIT_SOFTCAP=30.0
export QK_GAIN_INIT=1.0

# Encoder (bidirectional)
export ENC_DIM=256
export ENC_LAYERS=2
export ENC_HEADS=4

# Prior (causal over z)
export PRIOR_DIM=256
export PRIOR_LAYERS=3
export PRIOR_HEADS=4

# VAE (small segments + large latent for max info flow)
export SEGMENT_SIZE=16
export LATENT_DIM=384
export N_MEM_TOKENS=6
export KL_WEIGHT=1.0
export FREE_BITS=0.15
export KL_WARMUP_STEPS=8000
export USE_PRIOR_CONTEXT=1
export LATENT_STD_FLOOR=1e-4
export LATENT_RAW_STD_CLIP=8.0

# Training
export TRAIN_SEQ_LEN=1024
export TRAIN_BATCH_TOKENS=65536
export ITERATIONS=6000
export WARMDOWN_ITERS=3000
export WARMUP_STEPS=200
export MAX_WALLCLOCK_SECONDS=2400

# Optimizer
export EMBED_LR=0.03
export MATRIX_LR=0.02
export SCALAR_LR=0.01
export LATENT_LR=0.008
export MUON_MOMENTUM=0.95
export MUON_BACKEND_STEPS=5
export GRAD_CLIP_NORM=1.0

# EMA
export EMA_DECAY=0.997

# QAT (activates at 65% of training, resets EMA)
export QAT_FRACTION=0.65

# TTT (full-model SGD, momentum 0.95 per top submissions)
export TTT_ENABLED=1
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_MOMENTUM=0.95
export TTT_GRAD_CLIP=1.0
export TTT_LORA_RANK=0

# Logging
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=50

python3 train_gpt_vqvae.py

echo "Done! Log: logs/$RUN_ID.txt"
