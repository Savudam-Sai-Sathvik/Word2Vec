import torch
import numpy as np
from collections import Counter
import re
import os
import json
import argparse
from k_means_constrained import KMeansConstrained
from word2vec_finetuning import gen_embeddings


def clean_data(data):
    data = data.lower()
    p = r"\d+\.jpg\s+\(\d+[a-z]\)"
    data = re.sub(p, "", data)
    p = r'\b\w+\.jpg\b'
    data = re.sub(p, "", data)

    punctuation_counts = Counter(char for char in data if char in ".!?;:,—'")
    total_chars = len(data)
    total_no_punts = sum(punctuation_counts.values())
    punt_to_chunk_size = total_no_punts / total_chars if total_chars > 0 else 0.0

    sentences = re.split(r'(?<=[.!?])\s+', data)
    s = []
    for sen in sentences:
        clean_sen = re.sub(r'[^a-zA-Z\s]', '', sen)
        tokens = clean_sen.split()
        if tokens:
            s.append(tokens)

    return s, (punt_to_chunk_size, punctuation_counts[";"], punctuation_counts[","])


def sen_and_word_len(chunk_sentences):
    sen_lens = []
    word_lens = []
    for sentence in chunk_sentences:
        sen_lens.append(len(sentence))
        word_lens.extend(len(w) for w in sentence)
    if not sen_lens:
        return (0, 0), (0, 0)
    return (np.mean(sen_lens), np.var(sen_lens)), (np.mean(word_lens), np.var(word_lens))


def words_in_word2vec_voc(chunk_sentences, word2idx):
    return [word for sentence in chunk_sentences for word in sentence if word in word2idx]


def tfidf(voc_only_data, IDF):
    TF = {}
    unique_words = set()
    for i in range(len(voc_only_data)):
        chunk = voc_only_data[i]
        len_c = len(chunk)
        if len_c == 0:
            TF[i] = Counter()
            continue
        chunk_counter = Counter(chunk)
        for k in chunk_counter:
            chunk_counter[k] /= len_c
        TF[i] = chunk_counter
        unique_words.update(chunk_counter.keys())

    TFIDF = {}
    for doc, counter in TF.items():
        tfidf_counter = Counter()
        for word, term_freq in counter.items():
            tfidf_counter[word] = term_freq * IDF.get(word, 0.0)
        TFIDF[doc] = tfidf_counter
    return TFIDF


def weighted_embeddings(voc_words, TFIDF_counter, embeddings, word2index):
    embedding_dim = embeddings.shape[1]
    weighted_emb = np.zeros(embedding_dim)
    total_weight = 0
    for word in voc_words:
        if word not in word2index or word not in TFIDF_counter:
            continue
        idx = word2index[word]
        weighted_emb += TFIDF_counter[word] * embeddings[idx]
        total_weight += TFIDF_counter[word]
    # if total_weight > 0:
    #     weighted_emb /= total_weight
    return weighted_emb


def build_style_features(sen_len_feat, word_len_feat, punct_feat):
    return np.array([
        sen_len_feat[0], sen_len_feat[1],
        word_len_feat[0], word_len_feat[1],
        punct_feat[0], punct_feat[1], punct_feat[2]
    ], dtype=np.float32)


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def combine_features(semantic_emb, style_feat, alpha=0.85):
    semantic_emb = normalize(semantic_emb)
    style_feat = normalize(style_feat)
    return np.concatenate([alpha * semantic_emb, (1 - alpha) * style_feat])


def clustering(k, m, chunk_embeddings):
    clf = KMeansConstrained(n_clusters=k, size_min=m, random_state=42)
    return clf.fit_predict(chunk_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', default="models/word2vec_model.pt", type=str)
    args = parser.parse_args()


    base_ckpt = torch.load(args.model, weights_only=False)

    from word2vec_finetuning import gen_embeddings


    base_W = base_ckpt["word_embeddings"]
    base_C = base_ckpt["ctxt_embeddings"]
    base_word2index = base_ckpt["word2index"]
    base_word_scores = base_ckpt["word_scores"]
    base_idf = base_ckpt["IDF"]

    if isinstance(base_W, torch.Tensor):
        base_W = base_W.detach()
    if isinstance(base_C, torch.Tensor):
        base_C = base_C.detach()

    with open(args.input, 'r') as f:
        data = json.load(f)

    k = data["num_authors"]
    m = data["min_chunks_per_author"]
    chunks = data["chunks"]


    refined_data = {}
    for i, chunk in enumerate(chunks):
        c, _ = clean_data(chunk)
        refined_data[i] = c


    word2vec = gen_embeddings(
        refined_data,
        window=5,
        dyn_ctxt_wind=True,
        n_negs=5,
        min_count=3,
        finetuning=True
    )

 
    word2vec.loadword2vec(base_W, base_C, base_word2index, base_word_scores, base_idf)


    word2vec.idf = base_idf


    word2vec.prepare()
    word2vec.train(lr=0.005)

    ft_ckpt = torch.load("word2vec_finetuned.pt", weights_only=False)

    word_embeddings = ft_ckpt["finetuned_word_embeddings"].cpu().numpy()
    word2index = ft_ckpt["finetuned_word2index"]
    idf = ft_ckpt["finetuned_IDF"]

    with open(args.input, 'r') as f:
        data = json.load(f)

    k = data["num_authors"]
    m = data["min_chunks_per_author"]
    chunks = data["chunks"]

    cleaned_chunks = []
    punct_features = []
    for chunk in chunks:
        c, p = clean_data(chunk)
        cleaned_chunks.append(c)
        punct_features.append(p)

    voc_only_data = [words_in_word2vec_voc(c, word2index) for c in cleaned_chunks]
    tfidf_scores = tfidf(voc_only_data, idf)

    features = []
    for i in range(len(chunks)):
        sem = weighted_embeddings(voc_only_data[i], tfidf_scores[i], word_embeddings, word2index)
        sen_len_feat, word_len_feat = sen_and_word_len(cleaned_chunks[i])
        style = build_style_features(sen_len_feat, word_len_feat, punct_features[i])
        features.append(combine_features(sem, style, alpha=0.85))

    labels = clustering(k, m, np.array(features))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(labels.tolist(), f)
