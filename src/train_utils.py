"""
Training and validation loops.
"""
import torch
import wandb
from src.masks import make_padding_mask, make_tgt_mask


def train_epoch(model, loader, criterion, optimizer, scheduler,
                src_vocab, tgt_vocab, device, clip=1.0, log_interval=50):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, (src, tgt) in enumerate(loader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_padding_mask(src, src_vocab.pad_idx).to(device)
        tgt_mask = make_tgt_mask(tgt_inp, tgt_vocab.pad_idx).to(device)

        logits = model(src, tgt_inp, src_mask, tgt_mask)

        B, seq, vocab = logits.shape
        loss = criterion(logits.reshape(B * seq, vocab), tgt_out.reshape(B * seq))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Step scheduler BEFORE optimizer (sets the LR for this step)
        if scheduler is not None:
            scheduler.step()

        optimizer.step()   # ALWAYS call optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if batch_idx % log_interval == 0 and wandb.run is not None:
            wandb.log({
                'train/loss_step': loss.item(),
                'train/lr':        (scheduler.lr if scheduler else
                                    optimizer.param_groups[0]['lr']),
                'train/step':      scheduler.current_step if scheduler else batch_idx,
            })

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_epoch(model, loader, criterion, src_vocab, tgt_vocab, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_padding_mask(src, src_vocab.pad_idx).to(device)
        tgt_mask = make_tgt_mask(tgt_inp, tgt_vocab.pad_idx).to(device)

        logits = model(src, tgt_inp, src_mask, tgt_mask)
        B, seq, vocab = logits.shape
        loss = criterion(logits.reshape(B * seq, vocab), tgt_out.reshape(B * seq))
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)
