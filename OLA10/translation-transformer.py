#!/usr/bin/env python3
"""
Transformer ENG→POR Translation (fixed test_step)
Loads eng-por.txt locally, builds the ByteLevel BPE tokenizer exactly as in reference,
trains a Transformer encoder–decoder for 300 epochs, then tests without teacher forcing.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import urllib.request
import numpy as np
import tokenizers
import math
import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# -----------------------------------------------------------------------------
# DataModule: identical to reference but reads local eng-por.txt
# -----------------------------------------------------------------------------
class TokenizedTranslationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 100,
        n_sequences: int = 100000,
        val_split: float = 0.2,
        num_workers: int = 4,
        data_url: str = "eng-por.txt",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_sequences = n_sequences
        self.val_split = val_split
        self.num_workers = num_workers
        self.data_url = data_url
        self.tokenizer = tokenizers.ByteLevelBPETokenizer()
        self.src_timesteps = None
        self.dst_timesteps = None

    def setup(self, stage=None):
        # 1) load ENG–POR pairs from local file (or fallback to URL)
        data = []
        if os.path.exists(self.data_url):
            with open(self.data_url, "r", encoding="utf-8") as f:
                for line in f:
                    pair = line.strip().split("\t")[:2]
                    if len(pair) == 2:
                        data.append(pair)
        else:
            with urllib.request.urlopen(self.data_url) as raw_data:
                for line in raw_data:
                    pair = line.decode("utf-8").split("\t")[:2]
                    if len(pair) == 2:
                        data.append(pair)
        data = np.array(data)[: self.n_sequences]

        # 2) train BPE tokenizer
        src_texts = data[:, 0].tolist()
        dst_texts = data[:, 1].tolist()
        self.tokenizer.train_from_iterator(
            src_texts + dst_texts,
            vocab_size=30_000,
            min_frequency=2,
            special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"],
        )

        pad_id = self.tokenizer.token_to_id("<PAD>")
        sos_id = self.tokenizer.token_to_id("<SOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")

        # 3) encode and determine max lengths
        enc_src = self.tokenizer.encode_batch(src_texts)
        enc_dst = self.tokenizer.encode_batch(dst_texts)
        src_ids = [e.ids for e in enc_src]
        dst_ids = [e.ids for e in enc_dst]

        self.src_timesteps = max(len(ids) for ids in src_ids) + 2
        self.dst_timesteps = max(len(ids) for ids in dst_ids) + 2

        # 4) build dataset of (src_tensor, dst_tensor)
        def make_tensor(ids_list, timesteps):
            seq = [sos_id] + ids_list + [eos_id]
            if len(seq) < timesteps:
                seq += [pad_id] * (timesteps - len(seq))
            else:
                seq = seq[:timesteps]
            return torch.tensor(seq, dtype=torch.long)

        dataset = [
            (
                make_tensor(s, self.src_timesteps),
                make_tensor(d, self.dst_timesteps),
            )
            for s, d in zip(src_ids, dst_ids)
        ]

        # 5) split into train/val/test
        n_val = int(self.val_split * len(dataset))
        n_train = len(dataset) - n_val
        self.data_train, self.data_val = random_split(dataset, [n_train, n_val])
        self.data_test = self.data_val

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# -----------------------------------------------------------------------------
# Sinusoidal Positional Encoding (reference implementation)
# -----------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


# -----------------------------------------------------------------------------
# Transformer Model (reference implementation with fixed test_step)
# -----------------------------------------------------------------------------
class TransformerTranslationModel(pl.LightningModule):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.save_hyperparameters()
        vocab_size = 30_000
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=512)
        self.transformer = nn.Transformer(
            d_model, nhead, num_layers, num_layers, dim_feedforward=512, dropout=0.1
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, src, tgt):
        x = self.tok_emb(src) * math.sqrt(self.hparams.d_model)
        x = self.pos_enc(x)
        y = self.tok_emb(tgt) * math.sqrt(self.hparams.d_model)
        y = self.pos_enc(y)
        out = self.transformer(x.transpose(0, 1), y.transpose(0, 1))
        return self.fc_out(out).transpose(0, 1)

    def training_step(self, batch, _):
        src, tgt = batch
        inp, real = tgt[:, :-1], tgt[:, 1:]
        logits = self(src, inp)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), real.reshape(-1))
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        src, tgt = batch
        inp, real = tgt[:, :-1], tgt[:, 1:]
        logits = self(src, inp)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), real.reshape(-1))
        pred = logits.argmax(dim=-1)
        mask = real != 0
        acc = (pred == real).float()[mask].mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, _):
        src, tgt = batch
        batch_size, tgt_len = tgt.size()
        device = src.device
        sos_token = 1
        eos_token = 2

        # begin sequence
        generated = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)

        # greedy decode up to target length
        for _ in range(tgt_len - 1):
            logits = self(src, generated)  # (batch, seq_len, vocab_size)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token).all():
                break

        # truncate or pad to exactly tgt_len
        generated = generated[:, :tgt_len]

        # compute accuracy (ignore PAD=0)
        mask = tgt != 0
        acc = (generated == tgt).float()[mask].mean()
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main():
    pl.seed_everything(42)
    dm = TokenizedTranslationDataModule()
    model = TransformerTranslationModel()

    print("Parameter count:", sum(p.numel() for p in model.parameters()))
    logger = CSVLogger(".", name="translation-transformer")
    ckpt = ModelCheckpoint(monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=300,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=[ckpt],
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
