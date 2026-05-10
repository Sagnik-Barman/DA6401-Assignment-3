"""Mask creation utilities for Transformer."""
import torch


def make_padding_mask(seq, pad_idx):
    """
    Returns a bool mask of shape (B, 1, 1, seq_len).
    True → position is PAD and should be masked.
    """
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)   # (B,1,1,seq)


def make_causal_mask(seq_len, device):
    """
    Upper-triangular (look-ahead) causal mask.
    Shape: (1, 1, seq_len, seq_len).  True → masked.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)               # (1,1,seq,seq)


def make_tgt_mask(tgt, pad_idx):
    """
    Combined target mask: padding mask OR causal mask.
    Shape: (B, 1, seq_len, seq_len).
    """
    pad_mask   = make_padding_mask(tgt, pad_idx)        # (B,1,1,seq)
    causal_mask = make_causal_mask(tgt.size(1), tgt.device)  # (1,1,seq,seq)
    return pad_mask | causal_mask
