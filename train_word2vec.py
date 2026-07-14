import os
import re 
import numpy as np
from collections import Counter
import torch
import argparse
from tqdm import tqdm
import time as time

seed = 42
torch.manual_seed(seed)

class gen_embeddings:
    def __init__(self, data, dim=128, window=5, dyn_ctxt_wind=False, epochs=5, n_negs=5, min_count=3, finetuning=False, val_split=0.1):
        self.dim = dim
        self.alpha = 0.75
        self.dyn_ctx_wind = dyn_ctxt_wind
        self.word2index = {}
        self.data = data
        self.window = window
        self.n_negs = n_negs
        self.epochs = epochs
        self.val_split = val_split
     
        self.text_embed_size = 30000
        self.extra_embeds_size = 0
        self.train_dataset = []
        self.val_dataset = []  
        self.word_scores = []
        self.lr = 0.1
        self.min_count = min_count
        self.finetuning = finetuning
    
    def loadword2vec(self, W, C, word2index):
        self.W = W.clone().detach().requires_grad_(True)
        self.C = C.clone().detach().requires_grad_(True)
        self.word2index = word2index
        self.vocab_size = W.shape[0]

    def intialize(self):
        if (self.finetuning == False):
            self.vocab_size = self.text_embed_size + self.extra_embeds_size
            self.c_idxs = list(range(self.vocab_size))
            # Xavier initialization
            bound = np.sqrt(6.0 / (self.vocab_size + self.dim))
            self.W = torch.zeros(self.vocab_size, self.dim, requires_grad=True)
            self.W.data.uniform_(-bound, bound)
            
            self.C = torch.zeros(self.vocab_size, self.dim, requires_grad=True)
            self.C.data.uniform_(-bound, bound)
             
    def prepare_vocab_and_scores(self):
        all_words = []
        counters = []
        for author in self.data:
            author_words = np.concatenate(self.data[author]).tolist()
            counters.append(Counter(author_words))
            all_words.append(author_words)
        all_words = np.concatenate(all_words).tolist()
        word_counter = Counter(all_words)
        l = len(word_counter.keys())

        word_counter = {w: c for w, c in word_counter.items() if c >= self.min_count}
        unique_words_perdata = list(word_counter.keys())
        self.unique_words_size = len(unique_words_perdata)
        
        if (self.finetuning == False):
            self.vocab_words = unique_words_perdata
            self.text_embed_size = len(unique_words_perdata)
            print("size_before:", l, "reduced_size:", self.text_embed_size)
        
        corpus_size = sum(word_counter.values())
        IDF = {}
        N = len(self.data)
        for word in unique_words_perdata:
            count = 0
            for counter in counters:
                if(counter[word] > 0):
                    count += 1 
            IDF[word] = np.log((N+1)/(count+1)) 
        self.idf = IDF
         
        neg_sampling_p = []
        self.sub_p = {} 
        for i in range(self.unique_words_size):
            if (self.finetuning == False):
                self.word2index[unique_words_perdata[i]] = i 
            else:
                if unique_words_perdata[i] not in self.word2index:
                    continue
            n_wi = word_counter[unique_words_perdata[i]]
            p_alpha = (n_wi)**self.alpha
            neg_sampling_p.append(p_alpha)
            z_wi = n_wi/corpus_size
            p_keep = (np.sqrt(z_wi/0.001)+1)*(0.001/z_wi)
            idx = self.word2index[unique_words_perdata[i]]
            self.sub_p[idx] = min(1, p_keep)
        neg_sampling_p = np.array(neg_sampling_p, dtype=np.float32)
        self.word_scores = neg_sampling_p/np.sum(neg_sampling_p)

    def get_word_index(self, word):
        return self.word2index[word]

    def generate_context_window_pairs(self, sentence_tokens, author_id):
        total_sen_pairs = []
        sentence_indices = []
        for token in sentence_tokens:
            if token in self.word2index:
                idx = self.word2index[token]
                if np.random.rand() < self.sub_p[idx]:
                    sentence_indices.append(idx)
        N = len(sentence_indices)
        if N < 2:
            return []
        w = self.window
        if(self.dyn_ctx_wind):
            w = np.random.randint(1, self.window + 1)
        for centre in range(N):
            ctxt_pairs_idxs = []
            centre_idx = sentence_indices[centre]
            for j in range(max(0, centre-w), min(N, centre + w+1)):
                if centre != j:
                    ctxt_pairs_idxs.append((centre_idx, sentence_indices[j]))
            if ctxt_pairs_idxs:
                total_sen_pairs.append((ctxt_pairs_idxs, author_id))
        return total_sen_pairs
    
    def generate_train_dataset(self):
        all_pairs = []
        for author in range(len(self.data)):
            for sentence in self.data[author]:
                s = self.generate_context_window_pairs(sentence, author) 
                all_pairs.extend(s)
        
        # Split the dara into train and validation
        n_val = int(len(all_pairs) * self.val_split)
        indices = np.random.permutation(len(all_pairs))
        
        self.val_dataset = [all_pairs[i] for i in indices[:n_val]]
        self.train_dataset = [all_pairs[i] for i in indices[n_val:]]

        print(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        return self.train_dataset
    
    def data_loader(self, batch_size):
        indices = np.random.permutation(len(self.train_dataset))
        for i in range(0, len(indices), batch_size):
            data_batch = indices[i: i + batch_size]
            batch = [self.train_dataset[idx] for idx in data_batch]
            yield batch

    def calculate_loss(self, word_ctxt_idxs):
        word_idxs, ctxt_idxs = zip(*word_ctxt_idxs)
        word_idxs = list(word_idxs)
        ctxt_idxs = list(ctxt_idxs)
        N = len(word_idxs)
        
        w_embs = self.W[word_idxs]
        c_embs = self.C[ctxt_idxs]
        neg_word_idxs = torch.multinomial(self.word_scores, num_samples=N * self.n_negs, replacement=True).view(N, self.n_negs)
        nc_embs = self.C[neg_word_idxs]  
        
        scores = (w_embs * c_embs).sum(1)
        neg_scores = (w_embs.unsqueeze(1) * nc_embs).sum(2)
        
        pos_loss = -torch.log(torch.sigmoid(scores) + 1e-8).sum()
        neg_loss = -torch.log(torch.sigmoid(-neg_scores) + 1e-8).sum()
        
        # Normalize by N for stability
        # loss = (pos_loss + neg_loss) / N 
        loss = pos_loss + neg_loss
        return loss
         
    def train(self, lr=0.01, name="trained_model"):
        optimizer = torch.optim.Adam([self.W, self.C], lr)
        t1 = time.time()
        for i in tqdm(range(self.epochs)):
            batches = self.data_loader(batch_size=32)
            total_loss = 0
            num_batches = 0
            
            for batch in batches:
                loss = 0
                optimizer.zero_grad()
                for sentence in batch:
                    word_ctxt = sentence[0]
                    loss += self.calculate_loss(word_ctxt)
                loss = loss / len(batch)
                total_loss += loss.item()     
                loss.backward()
                optimizer.step()
                num_batches += 1
            if(i%2==0):
                    os.makedirs("models", exist_ok=True)
                    torch.save({
                        "word_embeddings": self.W.detach().cpu(),
                        "ctxt_embeddings": self.C.detach().cpu(),
                        "word2index": self.word2index,
                        "word_scores": self.word_scores,
                        "IDF": self.idf
                    }, f"models/word2vec_model.pt")
            # Validation
            val_loss = 0
            with torch.no_grad():
                for sentence in self.val_dataset:
                    word_ctxt = sentence[0]
                    val_loss += self.calculate_loss(word_ctxt).item()
            val_loss = val_loss / max(1, len(self.val_dataset))
            
            print(f"epoch: {i}, train_loss: {total_loss / max(1, num_batches):.4f}, val_loss: {val_loss:.4f}")
            
            os.makedirs("models", exist_ok=True)
            torch.save({
                "word_embeddings": self.W.detach().cpu(),
                "ctxt_embeddings": self.C.detach().cpu(),
                "word2index": self.word2index,
                "word_scores": self.word_scores,
                "IDF": self.idf
            }, f"models/{name}.pt")
            t2 = time.time()
            if(t2-t1)>1440:
                return True

        return True

    def prepare(self):
        t1 = time.time()
        self.prepare_vocab_and_scores()
        self.intialize()
        self.word_scores = torch.tensor(self.word_scores, dtype=torch.float32)
        self.generate_train_dataset()
        t2 = time.time()
        print("time_taken=", t2-t1)

    def embedding(self, word):
        idx = self.get_word_index(word)
        emb = self.W[idx]
        return emb


def txt2sen(data):
    data = data.lower()
    # Remove image info ex: 362.jpg (39k)
    p = r"\d+\.jpg\s+\(\d+[a-z]\)"
    data = re.sub(p, "", data)
    # Remove image info ex (cover.jpg)
    p = r'\b\w+\.jpg\b'
    data = re.sub(p, "", data)
    
    # Break the text into sentences with stopping punctuations
    sentences = re.split(r'(?<=[.!?])\s+', data)
    return sentences


def clean_data(txt_file_path):
    cleaned_author_data = {}
    for file in os.scandir(txt_file_path):
        with open(file) as f:
            data = f.read()
            filename = os.path.basename(f.name)
            label = int(filename.split('_')[1].split('.')[0])-1
            author_sentences = txt2sen(data)
            
            data = []
            for sentence in author_sentences:
                clean_sen = re.sub(r'[^a-zA-Z\s]', '', sentence)
                tokens = clean_sen.split()
                data.append(tokens)
            
            cleaned_author_data[label] = data
    return cleaned_author_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Word2Vec embeddings')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--dim', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--window', type=int, default=5, help='Context window size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--min_count', type=int, default=3, help='Minimum word count')
    parser.add_argument('--n_negs', type=int, default=5, help='Number of negative samples')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}...")
    refined_data = clean_data(args.data_path)
    
    print(f"Training Word2Vec with dim={args.dim}, window={args.window}, epochs={args.epochs}")
    word2vec = gen_embeddings(
        refined_data, 
        dim=args.dim,
        window=args.window,
        epochs=args.epochs,
        min_count=args.min_count,
        n_negs=args.n_negs
    )
    
    word2vec.prepare()
    word2vec.train(lr=args.lr, name="word2vec_model")
    
    print("Training complete! Model saved to models/word2vec_model.pt")
