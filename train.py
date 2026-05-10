"""
Main training script for DA6401 Assignment 3.
Usage:
    python train.py --config configs/base.yaml
    python train.py --wandb_project MY_PROJECT --epochs 20
"""
import argparse
import os
import yaml
import torch
import wandb

from src.data       import get_dataloaders
from src.model      import Transformer
from src.optim      import LabelSmoothingLoss, NoamScheduler
from src.train_utils import train_epoch, validate_epoch
from src.inference  import evaluate_bleu


# ─────────────────────────────────────────────────────────────
# Default hyperparameters (override via config file or CLI)
# ─────────────────────────────────────────────────────────────
DEFAULTS = dict(
    # Data
    batch_size   = 128,
    max_len      = 150,
    min_freq     = 2,
    num_workers  = 2,
    # Model
    d_model      = 256,
    num_heads    = 8,
    num_layers   = 3,
    d_ff         = 512,
    dropout      = 0.1,
    pos_encoding = 'sinusoidal',   # 'sinusoidal' | 'learned'
    # Training
    epochs       = 30,
    warmup_steps = 4000,
    label_smoothing = 0.1,
    clip         = 1.0,
    use_noam     = True,
    fixed_lr     = 1e-4,           # used only when use_noam=False
    # Scheduler factor
    noam_factor  = 1.0,
    # Logging
    wandb_project = 'da6401-assignment3',
    wandb_entity  = None,
    run_name      = 'base',
    log_interval  = 50,
    # Checkpoint
    save_path     = 'checkpoints/best_model.pt',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None else str
        if isinstance(v, bool):
            parser.add_argument(f'--{k}', action='store_true' if not v else 'store_false')
        else:
            parser.add_argument(f'--{k}', type=t, default=None)
    return parser.parse_args()


def merge_config(args):
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


def main():
    args = parse_args()
    cfg  = merge_config(args)

    os.makedirs(os.path.dirname(cfg['save_path']), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── W&B init ──────────────────────────────────────────────
    wandb.init(
        project = cfg['wandb_project'],
        entity  = cfg.get('wandb_entity'),
        name    = cfg['run_name'],
        config  = cfg,
    )

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size  = cfg['batch_size'],
        max_len     = cfg['max_len'],
        min_freq    = cfg['min_freq'],
        num_workers = cfg['num_workers'],
    )

    # ── Model ─────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = cfg['d_model'],
        num_heads      = cfg['num_heads'],
        num_layers     = cfg['num_layers'],
        d_ff           = cfg['d_ff'],
        max_len        = cfg['max_len'],
        dropout        = cfg['dropout'],
        pos_encoding   = cfg['pos_encoding'],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss ──────────────────────────────────────────────────
    criterion = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = tgt_vocab.pad_idx,
        smoothing  = cfg['label_smoothing'],
    )

    # ── Optimiser + Scheduler ─────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    if cfg['use_noam']:
        scheduler = NoamScheduler(optimizer, cfg['d_model'],
                                  cfg['warmup_steps'], cfg['noam_factor'])
    else:
        for pg in optimizer.param_groups:
            pg['lr'] = cfg['fixed_lr']
        scheduler = None

    # ── Training loop ─────────────────────────────────────────
    best_val_loss = float('inf')

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            src_vocab, tgt_vocab, device, cfg['clip'], cfg['log_interval']
        )
        val_loss = validate_epoch(model, val_loader, criterion, src_vocab, tgt_vocab, device)

        # BLEU every 5 epochs (expensive)
        val_bleu = None
        if epoch % 5 == 0 or epoch == cfg['epochs']:
            val_bleu = evaluate_bleu(model, val_loader, src_vocab, tgt_vocab, device)

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
              + (f" | val_bleu={val_bleu:.2f}" if val_bleu is not None else ""))

        log_dict = {
            'epoch':              epoch,
            'train/loss_epoch':   train_loss,
            'val/loss':           val_loss,
        }
        if val_bleu is not None:
            log_dict['val/bleu'] = val_bleu
        wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':      epoch,
                'model_state': model.state_dict(),
                'cfg':         cfg,
                'src_vocab':   src_vocab,
                'tgt_vocab':   tgt_vocab,
            }, cfg['save_path'])
            print(f"  ✓ Checkpoint saved (val_loss={val_loss:.4f})")

    # ── Final test BLEU ───────────────────────────────────────
    ckpt = torch.load(cfg['save_path'], map_location=device)
    model.load_state_dict(ckpt['model_state'])
    test_bleu = evaluate_bleu(model, test_loader, src_vocab, tgt_vocab, device)
    print(f"\nTest BLEU (best ckpt): {test_bleu:.2f}")
    wandb.log({'test/bleu': test_bleu})
    wandb.finish()


if __name__ == '__main__':
    main()
