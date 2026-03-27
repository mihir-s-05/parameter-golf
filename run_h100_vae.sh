#!/bin/bash
set -e

echo "=============================================="
echo "Segment-Bottleneck VAE LM — Single H100"
echo "=============================================="

pip install -q sentencepiece 2>/dev/null || true

export RUN_ID="sbvae_h100_$(date +%Y%m%d_%H%M%S)"
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export SEED=1337
export VOCAB_SIZE=1024

# Decoder
export MODEL_DIM=768
export DEC_LAYERS=7
export DEC_HEADS=12
export DEC_KV_HEADS=6
export DEC_MLP_MULT=3
export LOGIT_SOFTCAP=30.0
export QK_GAIN_INIT=1.5

# Encoder (bidirectional)
export ENC_DIM=384
export ENC_LAYERS=3
export ENC_HEADS=6

# Prior (causal over z)
export PRIOR_DIM=384
export PRIOR_LAYERS=4
export PRIOR_HEADS=6

# VAE
export SEGMENT_SIZE=32
export LATENT_DIM=192
export N_MEM_TOKENS=6
export KL_WEIGHT=1.0
export FREE_BITS=0.15
export KL_WARMUP_STEPS=2000

# Training
export TRAIN_SEQ_LEN=1024
export TRAIN_BATCH_TOKENS=65536
export ITERATIONS=25000
export WARMDOWN_ITERS=4000
export WARMUP_STEPS=200
export MAX_WALLCLOCK_SECONDS=5400

# Optimizer
export EMBED_LR=0.05
export MATRIX_LR=0.04
export SCALAR_LR=0.04
export MUON_MOMENTUM=0.95
export MUON_BACKEND_STEPS=5
export GRAD_CLIP_NORM=1.0

# EMA
export EMA_DECAY=0.997

# TTT
export TTT_ENABLED=1
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_MOMENTUM=0.9
export TTT_GRAD_CLIP=1.0

# Logging
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=50

python3 train_gpt_vqvae.py

echo "Done! Log: logs/$RUN_ID.txt"
