"""
Data loading for Multi30k (DE→EN) using spacy tokenisation.
"""
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy


# ─────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────
class Vocabulary:
    PAD = '<pad>'
    UNK = '<unk>'
    BOS = '<bos>'
    EOS = '<eos>'
    SPECIALS = [PAD, UNK, BOS, EOS]

    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = []
        for tok in self.SPECIALS:
            self._add(tok)

    def _add(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = len(self.idx2token)
            self.idx2token.append(token)

    def build(self, token_lists):
        from collections import Counter
        freq = Counter(tok for toks in token_lists for tok in toks)
        for token, count in freq.items():
            if count >= self.min_freq:
                self._add(token)

    def encode(self, tokens):
        unk = self.token2idx[self.UNK]
        return [self.token2idx.get(t, unk) for t in tokens]

    def decode(self, indices, skip_special=True):
        specials = set(self.SPECIALS) if skip_special else set()
        return [self.idx2token[i] for i in indices
                if self.idx2token[i] not in specials]

    def __len__(self):
        return len(self.idx2token)

    @property
    def pad_idx(self):  return self.token2idx[self.PAD]
    @property
    def unk_idx(self):  return self.token2idx[self.UNK]
    @property
    def bos_idx(self):  return self.token2idx[self.BOS]
    @property
    def eos_idx(self):  return self.token2idx[self.EOS]


# ─────────────────────────────────────────────────────────────
# Tokenisers
# ─────────────────────────────────────────────────────────────
def load_spacy_models():
    try:
        de_nlp = spacy.load('de_core_news_sm')
    except OSError:
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', 'de_core_news_sm'], check=True)
        de_nlp = spacy.load('de_core_news_sm')
    try:
        en_nlp = spacy.load('en_core_web_sm')
    except OSError:
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
        en_nlp = spacy.load('en_core_web_sm')
    return de_nlp, en_nlp


def tokenize(text, nlp):
    return [tok.text.lower() for tok in nlp.tokenizer(text.strip())]


# ─────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────
class Multi30kDataset:
    def __init__(self, hf_split, src_vocab, tgt_vocab, de_nlp, en_nlp, max_len=150):
        self.data      = hf_split
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.de_nlp    = de_nlp
        self.en_nlp    = en_nlp
        self.max_len   = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row   = self.data[idx]
        src_t = tokenize(row['de'], self.de_nlp)[:self.max_len]
        tgt_t = tokenize(row['en'], self.en_nlp)[:self.max_len]
        src_ids = [self.src_vocab.bos_idx] + self.src_vocab.encode(src_t) + [self.src_vocab.eos_idx]
        tgt_ids = [self.tgt_vocab.bos_idx] + self.tgt_vocab.encode(tgt_t) + [self.tgt_vocab.eos_idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


class CollateFn:
    """Top-level picklable collate class (required for Windows multiprocessing)."""
    def __init__(self, pad_src, pad_tgt):
        self.pad_src = pad_src
        self.pad_tgt = pad_tgt

    def __call__(self, batch):
        srcs, tgts = zip(*batch)
        src_padded = pad_sequence(srcs, batch_first=True, padding_value=self.pad_src)
        tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=self.pad_tgt)
        return src_padded, tgt_padded


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────
def get_dataloaders(batch_size=128, max_len=150, min_freq=2, num_workers=2):
    print("Loading Multi30k dataset …")
    raw = load_dataset('bentrevett/multi30k')

    de_nlp, en_nlp = load_spacy_models()

    def tok_de(texts): return [tokenize(t, de_nlp) for t in texts]
    def tok_en(texts): return [tokenize(t, en_nlp) for t in texts]

    train_split = raw['train']
    src_tokens  = tok_de(train_split['de'])
    tgt_tokens  = tok_en(train_split['en'])

    src_vocab = Vocabulary(min_freq=min_freq)
    tgt_vocab = Vocabulary(min_freq=min_freq)
    src_vocab.build(src_tokens)
    tgt_vocab.build(tgt_tokens)
    print(f"Src vocab size: {len(src_vocab)}  |  Tgt vocab size: {len(tgt_vocab)}")

    collate = CollateFn(src_vocab.pad_idx, tgt_vocab.pad_idx)

    def make_loader(split_name, shuffle):
        ds = Multi30kDataset(raw[split_name], src_vocab, tgt_vocab, de_nlp, en_nlp, max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate,
                          num_workers=0,
                          pin_memory=False)

    train_loader = make_loader('train',      shuffle=True)
    val_loader   = make_loader('validation', shuffle=False)
    test_loader  = make_loader('test',       shuffle=False)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
