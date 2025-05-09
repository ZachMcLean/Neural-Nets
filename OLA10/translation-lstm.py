#!/usr/bin/env python3
"""
Recurrent LSTM on Eng→Por Translation
Reuses the exact same TokenizedTranslationDataModule from translation-transformer.py.
"""

import os
import urllib.request
import numpy as np
import tokenizers
import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# -----------------------------------------------------------------------------
# Copy‑paste the exact TokenizedTranslationDataModule from translation-transformer.py
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

        enc_src = self.tokenizer.encode_batch(src_texts)
        enc_dst = self.tokenizer.encode_batch(dst_texts)
        src_ids = [e.ids for e in enc_src]
        dst_ids = [e.ids for e in enc_dst]

        self.src_timesteps = max(len(ids) for ids in src_ids) + 2
        self.dst_timesteps = max(len(ids) for ids in dst_ids) + 2

        def make_tensor(ids_list, timesteps):
            seq = [sos_id] + ids_list + [eos_id]
            if len(seq) < timesteps:
                seq += [pad_id] * (timesteps - len(seq))
            else:
                seq = seq[:timesteps]
            return torch.tensor(seq, dtype=torch.long)

        dataset = [
            (make_tensor(s, self.src_timesteps), make_tensor(d, self.dst_timesteps))
            for s, d in zip(src_ids, dst_ids)
        ]

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
# Encoder & Decoder with residual LSTMs (same as reference)
# -----------------------------------------------------------------------------
class ResidualLSTMEncoder(nn.Module):
    def __init__(self, vocab_size=30000, emb_dim=256, hid_dim=256, nlayers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, nlayers, batch_first=True, dropout=0.1
        )

    def forward(self, x):
        e = self.embedding(x)
        out, (h, c) = self.lstm(e)
        if out.shape == e.shape:
            out = out + e
        return out, (h, c)


class ResidualLSTMDecoder(nn.Module):
    def __init__(self, vocab_size=30000, emb_dim=256, hid_dim=256, nlayers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, nlayers, batch_first=True, dropout=0.1
        )
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x, hidden):
        e = self.embedding(x)
        out, hidden = self.lstm(e, hidden)
        if out.shape == e.shape:
            out = out + e
        return self.fc(out), hidden


class LSTMTranslationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = ResidualLSTMEncoder()
        self.decoder = ResidualLSTMDecoder()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, src, tgt):
        enc_out, hidden = self.encoder(src)
        logits, _ = self.decoder(tgt, hidden)
        return logits

    def training_step(self, batch, _):
        src, tgt = batch
        inp, real = tgt[:, :-1], tgt[:, 1:]
        logits = self(src, inp)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), real.reshape(-1))
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        src, tgt = batch
        inp, real = tgt[:, :-1], tgt[:, 1:]
        logits = self(src, inp)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), real.reshape(-1))
        pred = logits.argmax(dim=-1)
        mask = real != 0
        acc = (pred == real).float()[mask].mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, _):
        src, tgt = batch
        B, L = src.size()
        device = src.device
        gen = torch.full((B, 1), 1, dtype=torch.long, device=device)
        hidden = self.encoder(src)[1]
        for _ in range(L - 1):
            out, hidden = self.decoder(gen, hidden)
            nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=1)
        mask = tgt != 0
        acc = (gen[:, :L] == tgt).float()[mask].mean()
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)


# -----------------------------------------------------------------------------
def main():
    pl.seed_everything(42)
    dm = TokenizedTranslationDataModule()
    model = LSTMTranslationModel()

    print("Parameter count:", sum(p.numel() for p in model.parameters()))
    logger = CSVLogger(".", name="translation-lstm")
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
