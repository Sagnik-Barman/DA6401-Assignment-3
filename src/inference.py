"""
Inference utilities: greedy decoding and BLEU evaluation.
"""
import torch
from src.masks import make_padding_mask, make_causal_mask, make_tgt_mask


@torch.no_grad()
def greedy_decode(model, src, src_mask, tgt_vocab, max_len=100, device='cpu'):
    model.eval()
    B = src.size(0)
    enc_out = model.encode(src, src_mask)

    ys = torch.full((B, 1), tgt_vocab.bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        tgt_mask = make_causal_mask(ys.size(1), device)
        logits = model.decode(ys, enc_out, tgt_mask, src_mask)
        next_token = logits[:, -1, :].argmax(dim=-1)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished |= (next_token == tgt_vocab.eos_idx)
        if finished.all():
            break

    outputs = []
    for i in range(B):
        seq = ys[i, 1:].tolist()
        if tgt_vocab.eos_idx in seq:
            seq = seq[:seq.index(tgt_vocab.eos_idx)]
        outputs.append(seq)
    return outputs


@torch.no_grad()
def evaluate_bleu(model, data_loader, src_vocab, tgt_vocab, device):
    """Compute corpus-level BLEU score."""
    # Use sacrebleu directly to avoid naming conflict with local evaluate.py
    from sacrebleu.metrics import BLEU
    bleu_metric = BLEU(effective_order=True)

    model.eval()
    all_preds = []
    all_refs  = []

    for src, tgt in data_loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = make_padding_mask(src, src_vocab.pad_idx).to(device)

        pred_ids = greedy_decode(model, src, src_mask, tgt_vocab,
                                 max_len=tgt.size(1) + 10, device=device)

        for i in range(src.size(0)):
            pred_tokens = tgt_vocab.decode(pred_ids[i])
            ref_ids     = tgt[i].tolist()
            ref_tokens  = tgt_vocab.decode(ref_ids)
            all_preds.append(' '.join(pred_tokens))
            all_refs.append(' '.join(ref_tokens))

    result = bleu_metric.corpus_score(all_preds, [all_refs])
    return result.score   # already in percent


@torch.no_grad()
def translate_sentence(model, sentence_tokens, src_vocab, tgt_vocab, device, max_len=100):
    """Translate a single sentence (list of tokens)."""
    model.eval()
    src_ids = [src_vocab.bos_idx] + src_vocab.encode(sentence_tokens) + [src_vocab.eos_idx]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = make_padding_mask(src, src_vocab.pad_idx).to(device)
    pred_ids = greedy_decode(model, src, src_mask, tgt_vocab, max_len, device)
    return tgt_vocab.decode(pred_ids[0])
