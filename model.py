from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size,
                 maxlen=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        device = token_embedding.device
        return token_embedding + self.pos_embedding[:token_embedding.size(1), :].detach().to(device)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long())


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 num_decoder_layers,
                 emb_size,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward=512,
                 dropout=0.1,
                 activation="relu",
                 pad_idx_de=2,
                 pad_idx_en=2):
        super().__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       activation=activation,
                                       dropout=dropout,
                                       batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size, pad_idx_de)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size, pad_idx_en)
        self.positional_encoding = PositionalEncoding(emb_size)

    def forward(self,
                src,
                trg,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)


    @torch.inference_mode()
    def translate(self, x, sp_model_en, device="cuda:0"):
        x = torch.tensor(x)
        self.eval()
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        num_tokens = x.shape[1]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        max_len = num_tokens + 15
        start_symbol = sp_model_en.bos_id()
        x = x.to(device)
        src_mask = src_mask.to(device)
        memory = self.encode(x, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        memory = memory.to(device)
        for i in range(max_len-1):
            mask = (torch.triu(torch.ones((ys.shape[1], ys.shape[1]), device=device)) == 1).transpose(0, 1)
            tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            tgt_mask = tgt_mask.type(torch.bool).to(device)
            out = self.decode(ys, memory, tgt_mask)
            prob = self.generator(out[:, -1, :])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(x.data).fill_(next_word)], dim=1)
            if next_word == sp_model_en.eos_id():
                break
        tgt_tokens = ys.flatten()
        return sp_model_en.decode(tgt_tokens.cpu().tolist())


    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
