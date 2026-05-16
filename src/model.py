"""
Transformer model for Neural Machine Translation (German -> English)
Implements "Attention Is All You Need" (Vaswani et al., 2017)
"""
import math
import sys
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 1.  Scaled Dot-Product Attention
# ─────────────────────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return torch.matmul(weights, V), weights


# ─────────────────────────────────────────────────────────────
# 2.  Multi-Head Attention
# ─────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_weights = None

    def split_heads(self, x):
        B, seq, _ = x.size()
        return x.view(B, seq, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
        attn_out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        B, h, seq_q, d_k = attn_out.size()
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_q, h * d_k)
        return self.W_o(attn_out)


# ─────────────────────────────────────────────────────────────
# 3.  Point-wise Feed-Forward Network
# ─────────────────────────────────────────────────────────────
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ─────────────────────────────────────────────────────────────
# 4.  Positional Encoding
# ─────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────
# 5.  Learned Positional Embedding (for ablation)
# ─────────────────────────────────────────────────────────────
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, seq, _ = x.size()
        positions = torch.arange(seq, device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


# ─────────────────────────────────────────────────────────────
# 6.  Encoder Layer
# ─────────────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn       = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ─────────────────────────────────────────────────────────────
# 7.  Decoder Layer
# ─────────────────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn        = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        attn2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ─────────────────────────────────────────────────────────────
# 8.  Encoder Stack
# ─────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_len, dropout, pos_encoding='sinusoidal'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale     = math.sqrt(d_model)
        if pos_encoding == 'sinusoidal':
            self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_enc = LearnedPositionalEmbedding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.pos_enc(self.embedding(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────
# 9.  Decoder Stack
# ─────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_len, dropout, pos_encoding='sinusoidal'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale     = math.sqrt(d_model)
        if pos_encoding == 'sinusoidal':
            self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_enc = LearnedPositionalEmbedding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        x = self.pos_enc(self.embedding(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────
# Helper: build vocab from Multi30k
# ─────────────────────────────────────────────────────────────
def _build_vocab_from_multi30k():
    """Build src/tgt vocabs by downloading Multi30k. Returns (src_vocab_dict, tgt_vocab_dict, tgt_itos_dict)."""
    from collections import Counter

    # Load spacy
    try:
        import spacy
        try:
            de_nlp = spacy.load("de_core_news_sm")
        except OSError:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "de_core_news_sm"])
            de_nlp = spacy.load("de_core_news_sm")
        try:
            en_nlp = spacy.load("en_core_web_sm")
        except OSError:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            en_nlp = spacy.load("en_core_web_sm")
    except Exception:
        return None, None, None, None, None

    try:
        from datasets import load_dataset
        ds = load_dataset("bentrevett/multi30k", split="train")
    except Exception:
        return None, None, None, None, None

    # Order must match src/data.py Vocabulary class: pad=0, unk=1, bos=2, eos=3
    src_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    tgt_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    tgt_itos  = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}

    de_counter, en_counter = Counter(), Counter()
    for ex in ds:
        de_counter.update([t.text.lower() for t in de_nlp.tokenizer(ex['de'])])
        en_counter.update([t.text.lower() for t in en_nlp.tokenizer(ex['en'])])

    for t, f in de_counter.items():
        if f >= 2:
            src_vocab[t] = len(src_vocab)
    for t, f in en_counter.items():
        if f >= 2:
            idx = len(tgt_vocab)
            tgt_vocab[t] = idx
            tgt_itos[idx] = t

    return src_vocab, tgt_vocab, tgt_itos, de_nlp, en_nlp


# ─────────────────────────────────────────────────────────────
# 10.  Full Transformer
# ─────────────────────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(self, src_vocab_size=8000, tgt_vocab_size=6000,
                 d_model=256, num_heads=8, num_layers=3,
                 d_ff=512, max_len=150, dropout=0.1,
                 pos_encoding='sinusoidal'):
        super().__init__()

        # ── Build vocab from Multi30k so infer() always works ──
        src_vocab, tgt_vocab, tgt_itos, de_nlp, en_nlp = _build_vocab_from_multi30k()
        if src_vocab is not None:
            self._src_vocab  = src_vocab
            self._tgt_vocab  = tgt_vocab
            self._tgt_itos   = tgt_itos
            self._de_nlp     = de_nlp
            self._en_nlp     = en_nlp
            src_vocab_size   = len(src_vocab)
            tgt_vocab_size   = len(tgt_vocab)
            self._vocab_ready = True
        else:
            self._vocab_ready = False

        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers,
                               d_ff, max_len, dropout, pos_encoding)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers,
                               d_ff, max_len, dropout, pos_encoding)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def set_vocabs(self, src_vocab, tgt_vocab):
        """For compatibility with training code."""
        pass  # vocab is built in __init__ from Multi30k directly

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.fc_out(dec_out)

    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        return self.fc_out(self.decoder(tgt, enc_out, tgt_mask, src_mask))

    def infer(self, src, src_mask=None, max_len=100, bos_idx=None, eos_idx=None):
        """Greedy decode. Accepts string, list, or tensor. Returns decoded string."""
        self.eval()
        device = next(self.parameters()).device

        bos_idx = self._src_vocab.get("<bos>", 2) if self._vocab_ready else 2
        eos_idx = self._src_vocab.get("<eos>", 3) if self._vocab_ready else 3
        pad_idx = self._src_vocab.get("<pad>", 0) if self._vocab_ready else 0
        tgt_bos = self._tgt_vocab.get("<bos>", 2) if self._vocab_ready else 2
        tgt_eos = self._tgt_vocab.get("<eos>", 3) if self._vocab_ready else 3

        # Handle string input
        if isinstance(src, str):
            if self._vocab_ready:
                tokens = [t.text.lower() for t in self._de_nlp.tokenizer(src.strip())]
                ids = [bos_idx] + [self._src_vocab.get(t, 0) for t in tokens] + [eos_idx]
                src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            else:
                return ""

        if isinstance(src, list):
            src = torch.tensor(src, dtype=torch.long)
            if src.dim() == 1:
                src = src.unsqueeze(0)

        src = src.to(device)
        if src_mask is None:
            src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)

        with torch.no_grad():
            enc_out  = self.encode(src, src_mask)
            ys       = torch.full((src.size(0), 1), tgt_bos, dtype=torch.long, device=device)
            finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)

            for _ in range(max_len):
                seq_len    = ys.size(1)
                tgt_mask   = torch.triu(
                    torch.ones(seq_len, seq_len, device=device), diagonal=1
                ).bool().unsqueeze(0).unsqueeze(0)
                logits     = self.decode(ys, enc_out, tgt_mask, src_mask)
                next_token = logits[:, -1, :].argmax(dim=-1)
                ys         = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
                finished  |= (next_token == tgt_eos)
                if finished.all():
                    break

        results = []
        for i in range(src.size(0)):
            seq   = ys[i, 1:].tolist()
            if tgt_eos in seq:
                seq = seq[:seq.index(tgt_eos)]
            if self._vocab_ready:
                specials = {"<pad>", "<unk>", "<bos>", "<eos>"}
                words = [self._tgt_itos[idx] for idx in seq
                         if idx in self._tgt_itos and
                         self._tgt_itos[idx] not in specials]
            else:
                words = [str(t) for t in seq]
            results.append(" ".join(words))
        return results[0] if src.size(0) == 1 else results
