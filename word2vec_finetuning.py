import os
import re
import numpy as np
from collections import Counter
import torch
from tqdm import tqdm
import time

torch.manual_seed(42)


class gen_embeddings:
    def __init__(
        self,
        data,
        dim=100,
        window=5,
        dyn_ctxt_wind=False,
        epochs=5,
        n_negs=5,
        min_count=3,
        finetuning=False,
        val_split=0.1,
    ):
        self.data = data
        self.dim = dim
        self.window = window
        self.dyn_ctx_wind = dyn_ctxt_wind
        self.epochs = epochs
        self.n_negs = n_negs
        self.min_count = min_count
        self.val_split = val_split
        self.alpha = 0.75
        self.finetuning = finetuning

        self.word2index = {}
        self.word_scores = None
        self.sub_p = {}

        self.train_dataset = []
        self.val_dataset = []


    def loadword2vec(self, W, C, word2index, word_scores, idf):
        print("Loading BASE Word2Vec for fine-tuning")

        self.W_old = W.clone().detach()
        self.C_old = C.clone().detach()

        self.word2index = dict(word2index)
        self.old_vocab_size = W.shape[0]
        self.vocab_size = self.old_vocab_size

        self.word_scores_old = word_scores
        self.idf = idf

    def prepare_vocab_and_scores(self):
        all_words = []
        for author in self.data:
            all_words.extend(np.concatenate(self.data[author]).tolist())

        counter = Counter(all_words)
        counter = {w: c for w, c in counter.items() if c >= self.min_count}
        unique_words = list(counter.keys())

        if not self.finetuning:
            self.word2index = {w: i for i, w in enumerate(unique_words)}
            self.vocab_size = len(self.word2index)
        else:
            new_words = [w for w in unique_words if w not in self.word2index]
            self.new_vocab_size = len(new_words)

            for i, w in enumerate(new_words):
                self.word2index[w] = self.old_vocab_size + i

            self.vocab_size = self.old_vocab_size + self.new_vocab_size

     
        if self.finetuning:
            full_scores = np.ones(self.vocab_size, dtype=np.float32) * 1e-6
            full_scores[: self.old_vocab_size] = self.word_scores_old
        else:
            full_scores = np.zeros(self.vocab_size, dtype=np.float32)

        corpus_size = sum(counter.values())

        for w, c in counter.items():
            idx = self.word2index[w]
            full_scores[idx] = c ** self.alpha

            z = c / corpus_size
            p_keep = (np.sqrt(z / 0.001) + 1) * (0.001 / z)
            self.sub_p[idx] = min(1.0, p_keep)

        self.word_scores = full_scores / np.sum(full_scores)
        self.word_scores = torch.tensor(self.word_scores, dtype=torch.float32)

    def intialize(self):
        if not self.finetuning:
            bound = np.sqrt(6.0 / (self.vocab_size + self.dim))
            self.W = torch.empty(self.vocab_size, self.dim).uniform_(-bound, bound).requires_grad_()
            self.C = torch.empty(self.vocab_size, self.dim).uniform_(-bound, bound).requires_grad_()
        else:
            bound = np.sqrt(6.0 / self.dim)
            W_new = torch.empty(self.vocab_size - self.old_vocab_size, self.dim).uniform_(-bound, bound)
            C_new = torch.empty(self.vocab_size - self.old_vocab_size, self.dim).uniform_(-bound, bound)

            self.W = torch.cat([self.W_old, W_new], dim=0).requires_grad_()
            self.C = torch.cat([self.C_old, C_new], dim=0).requires_grad_()

    def generate_context_window_pairs(self, sentence):
        indices = []
        for token in sentence:
            if token in self.word2index:
                idx = self.word2index[token]
                if np.random.rand() < self.sub_p.get(idx, 1.0):
                    indices.append(idx)

        if len(indices) < 2:
            return []

        w = self.window
        if self.dyn_ctx_wind:
            w = np.random.randint(1, self.window + 1)

        pairs = []
        for i in range(len(indices)):
            for j in range(max(0, i - w), min(len(indices), i + w + 1)):
                if i != j:
                    pairs.append((indices[i], indices[j]))
        return pairs

    def generate_train_dataset(self):
        pairs = []
        for author in self.data:
            for sentence in self.data[author]:
                pairs.extend(self.generate_context_window_pairs(sentence))

        n_val = int(len(pairs) * self.val_split)
        perm = np.random.permutation(len(pairs))

        self.val_dataset = [pairs[i] for i in perm[:n_val]]
        self.train_dataset = [pairs[i] for i in perm[n_val:]]

        print(f"Train pairs: {len(self.train_dataset)}, Val pairs: {len(self.val_dataset)}")


    def calculate_loss(self, pairs):
        w_idx, c_idx = zip(*pairs)
        w_emb = self.W[list(w_idx)]
        c_emb = self.C[list(c_idx)]

        N = len(w_idx)
        neg_idx = torch.multinomial(self.word_scores, N * self.n_negs, replacement=True).view(N, self.n_negs)
        neg_emb = self.C[neg_idx]

        pos = (w_emb * c_emb).sum(1)
        neg = (w_emb.unsqueeze(1) * neg_emb).sum(2)

        return -torch.log(torch.sigmoid(pos) + 1e-8).sum() - torch.log(torch.sigmoid(-neg) + 1e-8).sum()

    def train(self, lr=0.005):
        optimizer = torch.optim.Adam([self.W, self.C], lr)

        for ep in tqdm(range(self.epochs)):
            np.random.shuffle(self.train_dataset)
            total_loss = 0

            for i in range(0, len(self.train_dataset), 32):
                batch = self.train_dataset[i : i + 32]
                optimizer.zero_grad()
                loss = self.calculate_loss(batch)
                loss.backward()

                if self.finetuning:
                    with torch.no_grad():
                        self.W.grad[: self.old_vocab_size] *= 0.1
                        self.C.grad[: self.old_vocab_size] *= 0.1

                optimizer.step()
                total_loss += loss.item()

            print(f"epoch {ep}, loss {total_loss / max(1, len(self.train_dataset)):.4f}")

        torch.save(
            {
                "finetuned_word_embeddings": self.W.detach().cpu(),
                "finetuned_ctxt_embeddings": self.C.detach().cpu(),
                "finetuned_word2index": self.word2index,
                "finetuned_word_scores": self.word_scores,
                "finetuned_IDF": self.idf,
            },
            "word2vec_finetuned.pt",
        )


    def prepare(self):
        self.prepare_vocab_and_scores()
        self.intialize()
        self.generate_train_dataset()
