"""
Transformer model for Neural Machine Translation (German -> English)
Implements "Attention Is All You Need" (Vaswani et al., 2017)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 1.  Scaled Dot-Product Attention
# ─────────────────────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q : (..., seq_q, d_k)
        K : (..., seq_k, d_k)
        V : (..., seq_k, d_v)
        mask : broadcastable bool tensor; True → position is masked
    Returns:
        output  : (..., seq_q, d_v)
        weights : (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)   # (..., seq_q, seq_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    # Replace NaN (all-masked rows) with 0 to avoid gradient issues
    weights = torch.nan_to_num(weights, nan=0.0)
    output = torch.matmul(weights, V)
    return output, weights


# ─────────────────────────────────────────────────────────────
# 2.  Multi-Head Attention
# ─────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        # x: (B, seq, d_model) → (B, h, seq, d_k)
        B, seq, _ = x.size()
        x = x.view(B, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        """
        query : (B, seq_q, d_model)
        key   : (B, seq_k, d_model)
        value : (B, seq_k, d_model)
        mask  : (B, 1, seq_q, seq_k) or (B, 1, 1, seq_k)  – True = masked
        """
        Q = self.split_heads(self.W_q(query))   # (B, h, seq_q, d_k)
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # mask already broadcastable across heads; add head dim if needed
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)             # (B,1,seq_q,seq_k)

        attn_out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: (B, h, seq_q, d_k)

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
# 4.  Positional Encoding  (sinusoidal, registered as buffer)
# ─────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                         # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                       # (1, max_len, d_model)
        self.register_buffer('pe', pe)                             # NOT a parameter

    def forward(self, x):
        # x: (B, seq, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────
# 5.  Learned Positional Embedding (for ablation 2.4)
# ─────────────────────────────────────────────────────────────
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, seq, _ = x.size()
        positions = torch.arange(seq, device=x.device).unsqueeze(0)  # (1, seq)
        return self.dropout(x + self.embedding(positions))


# ─────────────────────────────────────────────────────────────
# 6.  Encoder Layer
# ─────────────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    """Post-LayerNorm (as in the original paper)."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn       = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention + Add & Norm
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ─────────────────────────────────────────────────────────────
# 7.  Decoder Layer
# ─────────────────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads)   # masked
        self.cross_attn = MultiHeadAttention(d_model, num_heads)   # encoder-decoder
        self.ffn        = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        # Masked self-attention
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        # Cross-attention
        attn2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ─────────────────────────────────────────────────────────────
# 8.  Encoder Stack
# ─────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_len, dropout, pos_encoding='sinusoidal'):
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
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_len, dropout, pos_encoding='sinusoidal'):
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
# 10.  Full Transformer
# ─────────────────────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, num_heads=8, num_layers=3,
                 d_ff=512, max_len=256, dropout=0.1,
                 pos_encoding='sinusoidal'):
        super().__init__()
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

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.fc_out(dec_out)

    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.fc_out(dec_out)
