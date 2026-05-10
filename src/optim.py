"""
Loss function (label smoothing) and Noam learning-rate scheduler.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Label Smoothing Cross-Entropy Loss
# ─────────────────────────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing (ε).
    Ignores positions whose target == pad_idx.
    """
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        """
        logits  : (B*seq, vocab_size)
        targets : (B*seq,)
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            # Zero out pad positions so they don't contribute
            pad_mask = targets.eq(self.pad_idx)
            smooth_dist[pad_mask] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1)
        non_pad = (~pad_mask).float().sum()
        return loss.sum() / non_pad.clamp(min=1)


# ─────────────────────────────────────────────────────────────
# Noam Learning-Rate Scheduler
# ─────────────────────────────────────────────────────────────
class NoamScheduler:
    """
    lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    Usage: call .step() after every optimizer step (not after every epoch).
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer     = optimizer
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        self.factor        = factor
        self._step         = 0
        self._lr           = 0.0
        # Initialise to 0 so the first .step() sets the real lr
        self._set_lr(0.0)

    def _compute_lr(self, step):
        step = max(step, 1)
        return self.factor * (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def step(self):
        self._step += 1
        self._lr = self._compute_lr(self._step)
        self._set_lr(self._lr)

    @property
    def lr(self):
        return self._lr

    @property
    def current_step(self):
        return self._step
