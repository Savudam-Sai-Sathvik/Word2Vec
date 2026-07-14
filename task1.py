import json
import os 
import re
import numpy as np
from collections import Counter
import torch
import argparse


def sen_and_word_len(clean_data):
    sen_len_list = []
    word_len_list = []
    q_sen_lens = []
    q_word_lens = []
    for q_sen in clean_data["query_text"]:
        q_sen_lens.append(len(q_sen))         
        q_word_lens.extend(len(w) for w in q_sen)
    sen_len_list.append((np.mean(q_sen_lens), np.var(q_sen_lens)))
    word_len_list.append((np.mean(q_word_lens), np.var(q_word_lens)))
    for cand_sens in clean_data["candidates"]:
        c_sen_lens = []
        c_word_lens = []
        for c_sen in cand_sens:
            c_sen_lens.append(len(c_sen))
            c_word_lens.extend(len(w) for w in c_sen)
        sen_len_list.append((np.mean(c_sen_lens), np.var(c_sen_lens)))
        word_len_list.append((np.mean(c_word_lens), np.var(c_word_lens)))
    return sen_len_list, word_len_list


def words_in_word2vec_voc(clean_data, word2idx):
    voc_only_data = []
    q_sens = clean_data["query_text"]
    q_data = [word for q_sen in q_sens for word in q_sen if word in word2idx]
    voc_only_data.append(q_data)
    for cand in range(len(clean_data["candidates"])):
        c_data = []
        cand_sens = clean_data["candidates"][cand]
        c_data = [word for c_sen in cand_sens for word in c_sen if word in word2idx]
        voc_only_data.append(c_data)
    return voc_only_data


def tfidf(voc_only_data, IDF):
    TF = {}
    unique_words = set()
    for i in range(len(voc_only_data)):
        candidate = voc_only_data[i]
        len_c = len(candidate)
        cand_counter = Counter(candidate)
        for k in cand_counter:
            cand_counter[k] /= len_c
        TF[i] = cand_counter
        unique_words.update(cand_counter.keys())
    
    TFIDF = {}
    for doc, counter in TF.items():
        TFIDf_counter = Counter()
        for word, term_freq in counter.items():
            TFIDf_counter[word] = term_freq * IDF.get(word, 0)
        TFIDF[doc] = TFIDf_counter
    return TFIDF


def clean_data(data):
    data = data.lower()
    # Remove image info ex: 362.jpg (39k)
    p = r"\d+\.jpg\s+\(\d+[a-z]\)"
    data = re.sub(p, "", data)
    # Remove image info ex (cover.jpg)
    p = r'\b\w+\.jpg\b'
    data = re.sub(p, "", data)
    punctuation_counts = Counter(char for char in data if char in ".!?;:,—'")
    total_chars = len(data)
    total_no_punts = sum(punctuation_counts.values())
    punt_to_chunk_size = (
        total_no_punts / total_chars if total_chars > 0 else 0.0
    )
    # Break the text into sentences with stopping punctuations
    sentences = re.split(r'(?<=[.!?])\s+', data)
    s = []
    for sen in sentences:
        clean_sen = re.sub(r'[^a-zA-Z\s]', '', sen)
        tokens = clean_sen.split()
        s.append(tokens)
    return s, (punt_to_chunk_size, punctuation_counts[";"], punctuation_counts[","])


def weighted_embeddings(voc_only_data, TFIDF, embeddings, word2index):
    doc_emb = []
    embedding_dim = embeddings.shape[1]
    for i in range(len(voc_only_data)):
        weighted_emb = np.zeros(embedding_dim)
        total_weight = 0
        for word in voc_only_data[i]:
            idx = word2index[word]
            emb = embeddings[idx]
            tfidf_w = TFIDF[i][word] 
            weighted_emb += tfidf_w * emb
            total_weight += tfidf_w
        
        # # Normalize by total weight
        # if total_weight > 0:
        #     weighted_emb /= total_weight
        
        doc_emb.append(weighted_emb)

    return doc_emb


def build_style_features(sen_len_feat, word_len_feat, punct_feat):
    style = np.array([
        sen_len_feat[0], sen_len_feat[1],
        word_len_feat[0], word_len_feat[1],
        punct_feat[0], punct_feat[1], punct_feat[2]
    ], dtype=np.float32)              
    return style


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def combine_features(semantic_emb, style_feat, alpha=0.5):
    semantic_emb = normalize(semantic_emb)
    style_feat = normalize(style_feat)
    full_feature = np.concatenate([alpha * semantic_emb, (1 - alpha) * style_feat])
    return full_feature


def cosine(a, b, eps=1e-8):
    return np.dot(a, b) / ((np.linalg.norm(a)+eps)*(np.linalg.norm(b)+eps))


def ranking(data, punct_feat, doc_emb, sen_len_list, word_len_list):
    rankings = {}
    
    
    query_style = build_style_features(
        sen_len_list[0],
        word_len_list[0],
        punct_feat[0]
    )
    query_vec = combine_features(doc_emb[0], query_style, alpha=0.85)
    

    for i in range(1, len(doc_emb)): 
        cand_style = build_style_features(
            sen_len_list[i],
            word_len_list[i],
            punct_feat[i]
        )
        cand_vec = combine_features(doc_emb[i], cand_style, alpha=0.85)
        
   
        key_name = f"cand_{i}"
        rankings[key_name] = cosine(query_vec, cand_vec)
    
    rankings = dict(sorted(rankings.items(), key=lambda item: item[1], reverse=True))
    return list(rankings.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Task 1: Author Verification/Ranking')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--model',type=str,default= "models/word2vec_model.pt")

    
    args = parser.parse_args()
    
    checkpoint = torch.load(args.model, weights_only=False)
    embeddings = checkpoint["word_embeddings"]
    word2index = checkpoint["word2index"]
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    idf = checkpoint["IDF"]
    
    print(f"Loaded model from {args.model}")
    print(f"Vocabulary size: {len(word2index)}")
    

    with open(args.input, 'r') as f:
        data = json.load(f)
    
    data_punct_feat = []
    for i in range(len(data)):
        clean_cand = []
        punc_features = []
        clean_query, query_punct = clean_data(data[i]["query_text"])
        punc_features.append(query_punct)
        for cand in data[i]["candidates"]:
            clean_c, cand_punct = clean_data(data[i]["candidates"][cand])
            clean_cand.append(clean_c)
            punc_features.append(cand_punct)
        data[i]["query_text"] = clean_query
        data[i]["candidates"] = clean_cand 
        data_punct_feat.append(punc_features)       


    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        for i in range(len(data)):
            sen_len_list, word_len_list = sen_and_word_len(data[i])
            voc_only_data = words_in_word2vec_voc(data[i], word2index)
            t = tfidf(voc_only_data, idf)
            doc_emb = weighted_embeddings(voc_only_data, t, embeddings, word2index)
            
            # Rank candidates
            ranked = ranking(data[i], data_punct_feat[i], doc_emb, sen_len_list, word_len_list)
            
            out = {
                "query_id": f"q_{i}",
                "ranked_candidates": ranked
            }
            f.write(json.dumps(out) + "\n")
    
    print(f"Predictions saved to {args.output}")
