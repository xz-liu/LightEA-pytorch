import numpy as np
import os
from tqdm import tqdm
import faiss
import pickle
import json
from tqdm import tqdm, trange


def load_graph(path):
    if os.path.exists(path + "graph_cache.pkl"):
        return pickle.load(open(path + "graph_cache.pkl", "rb"))

    triples = []
    rel_size = 0
    with open(path + "triples_1", encoding='utf-8') as f:
        for line in f.readlines():
            h, r, t = [int(x) for x in line.strip().split("\t")]
            # print("load", h,r,t)
            triples.append([h, t, 2 * r])
            triples.append([t, h, 2 * r + 1])
            rel_size = max(rel_size, 2 * r + 1)
    with open(path + "triples_2", encoding='utf-8') as f:
        for line in f.readlines():
            h, r, t = [int(x) for x in line.strip().split("\t")]
            # print("load", h,r,t)
            triples.append([h, t, 2 * r])
            triples.append([t, h, 2 * r + 1])
            rel_size = max(rel_size, 2 * r + 1)
    triples = np.unique(triples, axis=0)
    print(f"rel_size is {rel_size}")
    node_size, rel_size = np.max(triples) + 1, np.max(triples[:, 2]) + 1
    print(f"node size is {node_size}, rel_size is {rel_size}")
    ent_tuple, triples_idx = [], []
    ent_ent_s, rel_ent_s, ent_rel_s = {}, set(), set()
    last, index = (-1, -1), -1

    for i in range(node_size):
        ent_ent_s[(i, i)] = 0

    for h, t, r in triples:
        ent_ent_s[(h, h)] += 1
        ent_ent_s[(t, t)] += 1

        if (h, t) != last:
            last = (h, t)
            index += 1
            ent_tuple.append([h, t])
            ent_ent_s[(h, t)] = 0

        triples_idx.append([index, r])
        ent_ent_s[(h, t)] += 1
        rel_ent_s.add((r, h))
        ent_rel_s.add((t, r))

    ent_tuple = np.array(ent_tuple)
    triples_idx = np.unique(np.array(triples_idx), axis=0)

    ent_ent = np.unique(np.array(list(ent_ent_s.keys())), axis=0)
    ent_ent_val = np.array([ent_ent_s[(x, y)] for x, y in ent_ent]).astype("float32")
    rel_ent = np.unique(np.array(list(rel_ent_s)), axis=0)
    ent_rel = np.unique(np.array(list(ent_rel_s)), axis=0)

    graph_data = [node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel]
    pickle.dump(graph_data, open(path + "graph_cache.pkl", "wb"))
    return graph_data


def load_aligned_pair(file_path, ratio=0.3):
    with open(file_path + "ref_ent_ids") as f:
        ref = f.readlines()
    try:
        with open(file_path + "sup_ent_ids") as f:
            sup = f.readlines()
    except:
        sup = None

    ref = np.array([list(map(int, line.replace("\n", "").split("\t"))) for line in ref]).astype(np.int64)
    if sup:
        sup = np.array([line.replace("\n", "").split("\t") for line in sup]).astype(np.int64)
        ref = np.concatenate([ref, sup])
    np.random.shuffle(ref)
    train_size = int(ref.shape[0] * ratio)

    print(ref.shape)
    return ref[:train_size], ref[train_size:]


def load_name_features(dataset, vector_path, mode="word-level"):
    try:
        word_vecs = pickle.load(open("./word_vectors.pkl", "rb"))
    except:
        word_vecs = {}
        with open(vector_path, encoding='UTF-8') as f:
            for line in tqdm(f.readlines()):
                line = line.split()
                word_vecs[line[0]] = [float(x) for x in line[1:]]
        pickle.dump(word_vecs, open("./word_vectors.pkl", "wb"))

    if "EN" in dataset:
        ent_names = json.load(open("translated_ent_name/%s.json" % dataset[:-1].lower(), "r"))

    d = {}
    count = 0
    for _, name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word) - 1):
                if word[idx:idx + 2] not in d:
                    d[word[idx:idx + 2]] = count
                    count += 1

    ent_vec = np.zeros((len(ent_names), 300), "float32")
    char_vec = np.zeros((len(ent_names), len(d)), "float32")
    for i, name in tqdm(ent_names):
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word) - 1):
                char_vec[i, d[word[idx:idx + 2]]] += 1
        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(d)) - 0.5

    faiss.normalize_L2(ent_vec)
    faiss.normalize_L2(char_vec)

    if mode == "word-level":
        name_feature = ent_vec
    if mode == "char-level":
        name_feature = char_vec
    if mode == "hybrid-level":
        name_feature = np.concatenate([ent_vec, char_vec], -1)

    return name_feature
