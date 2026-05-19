Github link - 
Wandb link - https://wandb.ai/ma25m024-indian-institute-of-technology-madras/da6401-assignment3/reports/Implementing-a-Transformer-for-Machine-Translation--VmlldzoxNjkzMTcyNw?accessToken=mr7qjplqq2hpsiswvlv1jgac4red1ixpqs3x69lcpuddmzompiltn777lqltj4mo

# DA6401 Assignment 3 – Neural Machine Translation with Transformers

Implementation of **"Attention Is All You Need"** (Vaswani et al., 2017) for German→English translation on the Multi30k dataset.

---

## Project Structure

```
da6401_assignment_3/
├── src/
│   ├── model.py          # Full Transformer (MHA, PE, Encoder, Decoder)
│   ├── masks.py          # Padding & causal mask utilities
│   ├── optim.py          # LabelSmoothingLoss + NoamScheduler
│   ├── data.py           # Multi30k loading, spacy tokenisation, Vocabulary
│   ├── train_utils.py    # train_epoch / validate_epoch
│   └── inference.py      # Greedy decoding + BLEU evaluation
├── configs/
│   └── base.yaml         # Default hyperparameters
├── train.py              # Main training script
├── ablations.py          # W&B ablation experiments (2.1–2.5)
├── evaluate.py           # Gradescope evaluation script
└── requirements.txt
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download spacy models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Training

```bash
# Train with default config (Noam scheduler, label smoothing ε=0.1)
python train.py --config configs/base.yaml --wandb_project YOUR_PROJECT --run_name base

# Override specific hyperparameters
python train.py --epochs 30 --d_model 256 --num_layers 3
```

---

## Ablation Experiments (W&B Report)

```bash
# 2.1 – Noam scheduler vs fixed LR
python ablations.py --exp noam_vs_fixed

# 2.2 – With vs without √(1/dk) scaling factor
python ablations.py --exp no_scale

# 2.3 – Attention head visualisation (run after training)
python ablations.py --exp attention_rollout --model_path checkpoints/best_model.pt

# 2.4 – Sinusoidal vs learned positional encoding
python ablations.py --exp learned_pos

# 2.5 – Label smoothing ε=0.1 vs ε=0.0
python ablations.py --exp label_smoothing

# Run all ablations sequentially
python ablations.py --exp all
```

---

## Evaluation

```bash
python evaluate.py --model_path checkpoints/best_model.pt
```

---

## Key Design Choices

| Component | Choice | Justification |
|---|---|---|
| Layer Norm | Post-LayerNorm | Matches the original paper exactly |
| Positional Encoding | Sinusoidal | Allows length extrapolation; no extra parameters |
| Label Smoothing | ε = 0.1 | Regularises overconfident predictions |
| LR Schedule | Noam (warmup=4000) | Prevents early divergence in attention layers |
| Tokenisation | spacy (`de_core_news_sm`, `en_core_web_sm`) | Required by assignment spec |

---

## Model Architecture

- **d_model** = 256, **num_heads** = 8, **num_layers** = 3, **d_ff** = 512
- Parameters: ~12M (appropriate for Multi30k)
- Attention: custom `scaled_dot_product_attention` + `MultiHeadAttention` (no `nn.MultiheadAttention`)
- Masks: separate padding mask and combined causal+padding target mask
