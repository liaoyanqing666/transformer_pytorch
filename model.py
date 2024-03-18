# writen by liaoyanqing666@github.com
import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    # Learnable positional encoding
    def __init__(self, emb_dim, len):
        super(LearnablePositionalEncoding, self).__init__()
        assert emb_dim > 0 and len > 0, 'emb_dim and len must be positive'
        self.emb_dim = emb_dim
        self.len = len
        self.pe = nn.Parameter(torch.zeros(len, emb_dim))

    def forward(self, x):
        return x + self.pe[:x.size(-2), :]


class PositionalEncoding(nn.Module):
    # Sine-cosine positional coding
    def __init__(self, emb_dim, max_len, freq=10000.0):
        super(PositionalEncoding, self).__init__()
        assert emb_dim > 0 and max_len > 0, 'emb_dim and max_len must be positive'
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.pe = torch.zeros(max_len, emb_dim)

        pos = torch.arange(0, max_len).unsqueeze(1)
        # pos: [max_len, 1]
        div = torch.pow(freq, torch.arange(0, emb_dim, 2) / emb_dim)
        # div: [ceil(emb_dim / 2)]
        self.pe[:, 0::2] = torch.sin(pos / div)
        # torch.sin(pos / div): [max_len, ceil(emb_dim / 2)]
        self.pe[:, 1::2] = torch.cos(pos / (div if emb_dim % 2 == 0 else div[:-1]))
        # torch.cos(pos / div): [max_len, floor(emb_dim / 2)]

    def forward(self, x, len=None):
        if len is None:
            len = x.size(-2)
        return x + self.pe[:len, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, dim_qk=None, dim_v=None, num_heads=1, dropout=0.):
        super(MultiHeadAttention, self).__init__()

        dim_qk = dim if dim_qk is None else dim_qk
        dim_v = dim if dim_v is None else dim_v

        assert dim % num_heads == 0 and dim_v % num_heads == 0 and dim_qk % num_heads == 0, 'dim must be divisible by num_heads'

        self.dim = dim
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(dim, dim_qk)
        self.w_k = nn.Linear(dim, dim_qk)
        self.w_v = nn.Linear(dim, dim_v)

    def forward(self, q, k, v, mask=None):
        # q: [B, len_q, D]
        # k: [B, len_kv, D]
        # v: [B, len_kv, D]
        assert q.ndim == k.ndim == v.ndim == 3, 'input must be 3-dimensional'

        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)
        assert q.size(-1) == k.size(-1) == v.size(-1) == self.dim, 'dimension mismatch'
        assert len_k == len_v, 'len_k and len_v must be equal'
        len_kv = len_v

        q = self.w_q(q).view(-1, len_q, self.num_heads, self.dim_qk // self.num_heads)
        k = self.w_k(k).view(-1, len_kv, self.num_heads, self.dim_qk // self.num_heads)
        v = self.w_v(v).view(-1, len_kv, self.num_heads, self.dim_v // self.num_heads)
        # q: [B, len_q, num_heads, dim_qk//num_heads]
        # k: [B, len_kv, num_heads, dim_qk//num_heads]
        # v: [B, len_kv, num_heads, dim_v//num_heads]
        # The following 'dim_(qk)//num_heads' is writen as d_(qk)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q: [B, num_heads, len_q, d_qk]
        # k: [B, num_heads, len_kv, d_qk]
        # v: [B, num_heads, len_kv, d_v]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_qk ** 0.5)
        # attn: [B, num_heads, len_q, len_kv]

        if mask is not None:
            attn = attn.transpose(0, 1).masked_fill(mask, float('-1e20')).transpose(0, 1)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        # output: [B, num_heads, len_q, d_v]
        output = output.transpose(1, 2)
        # output: [B, len_q, num_heads, d_v]
        output = output.contiguous().view(-1, len_q, self.dim_v)
        # output: [B, len_q, num_heads * d_v] = [B, len_q, dim_v]
        return output


class Feedforward(nn.Module):
    def __init__(self, dim, hidden_dim=2048, dropout=0., activate=nn.ReLU()):
        super(Feedforward, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = activate

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def attn_mask(len):
    """
    :param len: length of sequence
    :return: mask tensor, False for not replaced, True for replaced as -inf
    e.g. attn_mask(3) =
        tensor([[[False,  True,  True],
                 [False, False,  True],
                 [False, False, False]]])
    """
    mask = torch.triu(torch.ones(len, len, dtype=torch.bool), 1)
    return mask


def padding_mask(pad_q, pad_k):
    """
    :param pad_q: pad label of query (0 is padding, 1 is not padding), [B, len_q]
    :param pad_k: pad label of key (0 is padding, 1 is not padding), [B, len_k]
    :return: mask tensor, False for not replaced, True for replaced as -inf

    e.g. pad_q = tensor([[1, 1, 0]], [1, 0, 1])
        padding_mask(pad_q, pad_q) =
        tensor([[[False, False,  True],
                 [False, False,  True],
                 [ True,  True,  True]],

                [[False,  True, False],
                 [ True,  True,  True],
                 [False,  True, False]]])

    """
    assert pad_q.ndim == pad_k.ndim == 2, 'pad_q and pad_k must be 2-dimensional'
    assert pad_q.size(0) == pad_k.size(0), 'batch size mismatch'
    mask = pad_q.bool().unsqueeze(2) * pad_k.bool().unsqueeze(1)
    mask = ~mask
    # mask: [B, len_q, len_k]
    return mask


class EncoderLayer(nn.Module):
    def __init__(self, dim, dim_qk=None, num_heads=1, dropout=0., pre_norm=False):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(dim, dim_qk=dim_qk, num_heads=num_heads, dropout=dropout)
        self.ffn = Feedforward(dim, dim * 4, dropout)
        self.pre_norm = pre_norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        if self.pre_norm:
            res1 = self.norm1(x)
            x = x + self.attn(res1, res1, res1, mask)
            res2 = self.norm2(x)
            x = x + self.ffn(res2)
        else:
            x = self.attn(x, x, x, mask) + x
            x = self.norm1(x)
            x = self.ffn(x) + x
            x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, dim, dim_qk=None, num_heads=1, num_layers=1, dropout=0., pre_norm=False):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim, dim_qk, num_heads, dropout, pre_norm) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim, dim_qk=None, num_heads=1, dropout=0., pre_norm=False):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttention(dim, dim_qk=dim_qk, num_heads=num_heads, dropout=dropout)
        self.attn2 = MultiHeadAttention(dim, dim_qk=dim_qk, num_heads=num_heads, dropout=dropout)
        self.ffn = Feedforward(dim, dim * 4, dropout)
        self.pre_norm = pre_norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, enc, self_mask=None, pad_mask=None):
        if self.pre_norm:
            res1 = self.norm1(x)
            x = x + self.attn1(res1, res1, res1, self_mask)
            res2 = self.norm2(x)
            x = x + self.attn2(res2, enc, enc, pad_mask)
            res3 = self.norm3(x)
            x = x + self.ffn(res3)
        else:
            x = self.attn1(x, x, x, self_mask) + x
            x = self.norm1(x)
            x = self.attn2(x, enc, enc, pad_mask) + x
            x = self.norm2(x)
            x = self.ffn(x) + x
            x = self.norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim, dim_qk=None, num_heads=1, num_layers=1, dropout=0., pre_norm=False):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim, dim_qk, num_heads, dropout, pre_norm) for _ in range(num_layers)])

    def forward(self, x, enc, self_mask=None, pad_mask=None):
        for layer in self.layers:
            x = layer(x, enc, self_mask, pad_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, vocabulary, num_heads=1, num_layers=1, dropout=0., learnable_pos=False, pre_norm=False):
        super(Transformer, self).__init__()
        self.dim = dim
        self.vocabulary = vocabulary
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learnable_pos = learnable_pos
        self.pre_norm = pre_norm

        self.embedding = nn.Embedding(vocabulary, dim)
        self.pos_enc = LearnablePositionalEncoding(dim, 100) if learnable_pos else PositionalEncoding(dim, 100)
        self.encoder = Encoder(dim, dim // num_heads, num_heads, num_layers, dropout, pre_norm)
        self.decoder = Decoder(dim, dim // num_heads, num_heads, num_layers, dropout, pre_norm)
        self.linear = nn.Linear(dim, vocabulary)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, pad_mask=None):
        src = self.embedding(src)
        src = self.pos_enc(src)
        src = self.encoder(src, src_mask)

        tgt = self.embedding(tgt)
        tgt = self.pos_enc(tgt)
        tgt = self.decoder(tgt, src, tgt_mask, pad_mask)

        output = self.linear(tgt)
        return output

    def get_mask(self, tgt, src_pad=None):
        # Under normal circumstances, tgt_pad will perform mask processing when calculating loss, and it isn't necessarily in decoder
        if src_pad is not None:
            src_mask = padding_mask(src_pad, src_pad)
        else:
            src_mask = None
        tgt_mask = attn_mask(tgt.size(1))
        if src_pad is not None:
            pad_mask = padding_mask(torch.zeros_like(tgt), src_pad)
        else:
            pad_mask = None
        # src_mask: [B, len_src, len_src]
        # tgt_mask: [len_tgt, len_tgt]
        # pad_mask: [B, len_tgt, len_src]
        return src_mask, tgt_mask, pad_mask


if __name__ == '__main__':
    model = Transformer(512, 10000, 8, 6, 0.1, True, True)
    src = torch.randint(0, 10000, (2, 10))
    tgt = torch.randint(0, 10000, (2, 8))
    src_pad = torch.randint(0, 2, (2, 10))
    src_mask, tgt_mask, pad_mask = model.get_mask(tgt, src_pad)
    print(model(src, tgt, src_mask, tgt_mask, pad_mask).size())
    # torch.Size([2, 8, 10000])
