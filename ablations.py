"""
Ablation experiments for the W&B report.
Run: python ablations.py --exp <name>

Experiments:
  noam_vs_fixed       – 2.1  Noam vs fixed LR
  no_scale            – 2.2  Scaling factor ablation (logs grad norms)
  learned_pos         – 2.4  Learned vs sinusoidal PE
  label_smoothing     – 2.5  ε=0.1 vs ε=0.0
"""
import argparse
import os
import torch
import wandb
import numpy as np

from src.data        import get_dataloaders
from src.model       import Transformer
from src.optim       import LabelSmoothingLoss, NoamScheduler
from src.train_utils import train_epoch, validate_epoch
from src.inference   import evaluate_bleu
from src.masks       import make_padding_mask, make_tgt_mask


BASE_CFG = dict(
    batch_size=128, max_len=150, min_freq=2, num_workers=2,
    d_model=256, num_heads=8, num_layers=3, d_ff=512, dropout=0.1,
    epochs=20, warmup_steps=4000, label_smoothing=0.1,
    clip=1.0, wandb_project='da6401-assignment3',
)


def build_model(src_vocab, tgt_vocab, cfg, pos_encoding='sinusoidal'):
    return Transformer(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        d_model=cfg['d_model'], num_heads=cfg['num_heads'],
        num_layers=cfg['num_layers'], d_ff=cfg['d_ff'],
        max_len=cfg['max_len'], dropout=cfg['dropout'],
        pos_encoding=pos_encoding,
    )


def run_training(model, train_loader, val_loader, src_vocab, tgt_vocab,
                 cfg, scheduler_obj, criterion, device, extra_hooks=None):
    """Generic training loop returning (train_losses, val_losses)."""
    train_losses, val_losses = [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    sched = scheduler_obj(optimizer) if callable(scheduler_obj) else scheduler_obj

    for epoch in range(1, cfg['epochs'] + 1):
        tl = train_epoch(model, train_loader, criterion, optimizer, sched,
                         src_vocab, tgt_vocab, device, cfg['clip'], log_interval=50)
        vl = validate_epoch(model, val_loader, criterion, src_vocab, tgt_vocab, device)
        train_losses.append(tl)
        val_losses.append(vl)

        log = {'epoch': epoch, 'train_loss': tl, 'val_loss': vl}
        if epoch % 5 == 0 or epoch == cfg['epochs']:
            bleu = evaluate_bleu(model, val_loader, src_vocab, tgt_vocab, device)
            log['val_bleu'] = bleu
        if extra_hooks:
            log.update(extra_hooks(epoch, model))
        wandb.log(log)
        print(f"  epoch {epoch:3d} | train={tl:.4f} | val={vl:.4f}")
    return train_losses, val_losses


# ─────────────────────────────────────────────────────────────
# 2.1  Noam vs Fixed LR
# ─────────────────────────────────────────────────────────────
def exp_noam_vs_fixed(data, device):
    train_loader, val_loader, _, src_vocab, tgt_vocab = data
    criterion = LabelSmoothingLoss(len(tgt_vocab), tgt_vocab.pad_idx, 0.1)

    for variant, use_noam in [('noam', True), ('fixed_lr', False)]:
        wandb.init(project=BASE_CFG['wandb_project'],
                   name=f'ablation_noam_{variant}', config={**BASE_CFG, 'variant': variant},
                   reinit=True)
        model = build_model(src_vocab, tgt_vocab, BASE_CFG).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        if use_noam:
            sched = NoamScheduler(optimizer, BASE_CFG['d_model'], BASE_CFG['warmup_steps'])
        else:
            for pg in optimizer.param_groups: pg['lr'] = 1e-4
            sched = None
        run_training(model, train_loader, val_loader, src_vocab, tgt_vocab,
                     BASE_CFG, lambda opt: sched, criterion, device)
        wandb.finish()


# ─────────────────────────────────────────────────────────────
# 2.2  Scaling factor ablation + gradient norms
# ─────────────────────────────────────────────────────────────
def exp_scaling_factor(data, device):
    """
    Monkey-patches scaled_dot_product_attention to skip sqrt(d_k).
    Logs gradient norms of Q and K weight matrices for first 1000 steps.
    """
    import src.model as model_module

    train_loader, val_loader, _, src_vocab, tgt_vocab = data
    criterion = LabelSmoothingLoss(len(tgt_vocab), tgt_vocab.pad_idx, 0.1)

    original_sdpa = model_module.scaled_dot_product_attention

    def no_scale_sdpa(Q, K, V, mask=None):
        import torch.nn.functional as F
        scores = torch.matmul(Q, K.transpose(-2, -1))        # NO sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.matmul(weights, V), weights

    for variant, patch in [('with_scale', False), ('no_scale', True)]:
        if patch:
            model_module.scaled_dot_product_attention = no_scale_sdpa
        else:
            model_module.scaled_dot_product_attention = original_sdpa

        wandb.init(project=BASE_CFG['wandb_project'],
                   name=f'ablation_scale_{variant}',
                   config={**BASE_CFG, 'variant': variant}, reinit=True)

        model = build_model(src_vocab, tgt_vocab, BASE_CFG).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        sched = NoamScheduler(optimizer, BASE_CFG['d_model'], BASE_CFG['warmup_steps'])

        step = [0]
        orig_step = sched.step

        def logged_step():
            orig_step()
            if step[0] < 1000:
                # Log Q/K grad norms (after loss.backward but before optimizer.step)
                q_norms, k_norms = [], []
                for layer in model.encoder.layers:
                    if layer.self_attn.W_q.weight.grad is not None:
                        q_norms.append(layer.self_attn.W_q.weight.grad.norm().item())
                        k_norms.append(layer.self_attn.W_k.weight.grad.norm().item())
                if q_norms:
                    wandb.log({'grad/q_norm': np.mean(q_norms),
                               'grad/k_norm': np.mean(k_norms),
                               'step': sched.current_step})
            step[0] += 1

        sched.step = logged_step

        run_training(model, train_loader, val_loader, src_vocab, tgt_vocab,
                     {**BASE_CFG, 'epochs': 10}, lambda opt: sched, criterion, device)
        wandb.finish()

    model_module.scaled_dot_product_attention = original_sdpa   # restore


# ─────────────────────────────────────────────────────────────
# 2.3  Attention Rollout (run after main training)
# ─────────────────────────────────────────────────────────────
def exp_attention_rollout(model_path, data, device):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.masks import make_padding_mask

    train_loader, val_loader, _, src_vocab, tgt_vocab = data
    ckpt  = torch.load(model_path, map_location=device)
    cfg   = ckpt['cfg']

    model = build_model(src_vocab, tgt_vocab, cfg).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Pick first batch
    src, tgt = next(iter(val_loader))
    src = src[:1].to(device)
    src_mask = make_padding_mask(src, src_vocab.pad_idx).to(device)

    with torch.no_grad():
        _ = model.encode(src, src_mask)
        # Collect attention weights from the last encoder layer
        attn_weights = model.encoder.layers[-1].self_attn.attn_weights  # (1, h, seq, seq)

    tokens = src_vocab.decode(src[0].tolist(), skip_special=False)

    wandb.init(project=cfg.get('wandb_project', 'da6401-assignment3'),
               name='attention_rollout', reinit=True)

    num_heads = attn_weights.size(1)
    fig, axes = plt.subplots(2, num_heads // 2, figsize=(4 * num_heads // 2, 8))
    axes = axes.flatten()
    for h in range(num_heads):
        w = attn_weights[0, h].cpu().numpy()
        ax = axes[h]
        im = ax.imshow(w, cmap='viridis')
        ax.set_title(f'Head {h+1}')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticklabels(tokens, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = '/tmp/attention_heads.png'
    plt.savefig(fig_path, dpi=100)
    wandb.log({'attention/head_heatmaps': wandb.Image(fig_path)})
    plt.close()
    wandb.finish()
    print("Attention rollout logged to W&B.")


# ─────────────────────────────────────────────────────────────
# 2.4  Sinusoidal vs Learned Positional Encoding
# ─────────────────────────────────────────────────────────────
def exp_pos_encoding(data, device):
    train_loader, val_loader, _, src_vocab, tgt_vocab = data
    criterion = LabelSmoothingLoss(len(tgt_vocab), tgt_vocab.pad_idx, 0.1)

    for variant in ['sinusoidal', 'learned']:
        wandb.init(project=BASE_CFG['wandb_project'],
                   name=f'ablation_pos_{variant}',
                   config={**BASE_CFG, 'pos_encoding': variant}, reinit=True)
        model = build_model(src_vocab, tgt_vocab, BASE_CFG, pos_encoding=variant).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        sched = NoamScheduler(optimizer, BASE_CFG['d_model'], BASE_CFG['warmup_steps'])
        run_training(model, train_loader, val_loader, src_vocab, tgt_vocab,
                     BASE_CFG, lambda opt: sched, criterion, device)
        wandb.finish()


# ─────────────────────────────────────────────────────────────
# 2.5  Label Smoothing ablation
# ─────────────────────────────────────────────────────────────
def exp_label_smoothing(data, device):
    train_loader, val_loader, _, src_vocab, tgt_vocab = data

    for eps in [0.0, 0.1]:
        criterion = LabelSmoothingLoss(len(tgt_vocab), tgt_vocab.pad_idx, eps)
        wandb.init(project=BASE_CFG['wandb_project'],
                   name=f'ablation_ls_eps{eps}',
                   config={**BASE_CFG, 'label_smoothing': eps}, reinit=True)
        model = build_model(src_vocab, tgt_vocab, BASE_CFG).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        sched = NoamScheduler(optimizer, BASE_CFG['d_model'], BASE_CFG['warmup_steps'])

        # Extra hook: log prediction confidence (softmax prob of correct token)
        def conf_hook(epoch, m):
            if epoch % 5 != 0: return {}
            m.eval()
            conf_vals = []
            with torch.no_grad():
                for src, tgt in train_loader:
                    src = src.to(device); tgt = tgt.to(device)
                    tgt_inp = tgt[:, :-1]; tgt_out = tgt[:, 1:]
                    from src.masks import make_padding_mask, make_tgt_mask
                    sm = make_padding_mask(src, src_vocab.pad_idx).to(device)
                    tm = make_tgt_mask(tgt_inp, tgt_vocab.pad_idx).to(device)
                    logits = m(src, tgt_inp, sm, tm)
                    probs  = torch.softmax(logits, dim=-1)
                    # Gather correct-token probabilities
                    correct_probs = probs.gather(2, tgt_out.unsqueeze(-1)).squeeze(-1)
                    pad_mask = tgt_out != tgt_vocab.pad_idx
                    conf_vals.append(correct_probs[pad_mask].mean().item())
                    if len(conf_vals) > 10: break
            m.train()
            return {'train/pred_confidence': float(np.mean(conf_vals))}

        run_training(model, train_loader, val_loader, src_vocab, tgt_vocab,
                     BASE_CFG, lambda opt: sched, criterion, device,
                     extra_hooks=conf_hook)
        wandb.finish()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True,
                        choices=['noam_vs_fixed', 'no_scale', 'attention_rollout',
                                 'learned_pos', 'label_smoothing', 'all'])
    parser.add_argument('--model_path', default='checkpoints/best_model.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data = get_dataloaders(batch_size=128, max_len=150, min_freq=2, num_workers=2)

    if args.exp in ('noam_vs_fixed', 'all'):
        exp_noam_vs_fixed(data, device)
    if args.exp in ('no_scale', 'all'):
        exp_scaling_factor(data, device)
    if args.exp in ('attention_rollout', 'all'):
        exp_attention_rollout(args.model_path, data, device)
    if args.exp in ('learned_pos', 'all'):
        exp_pos_encoding(data, device)
    if args.exp in ('label_smoothing', 'all'):
        exp_label_smoothing(data, device)
