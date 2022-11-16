# choose the GPU, "-1" represents using the CPU

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import pickle
# import all the requirements
import faiss
from utils import *
from tqdm import tqdm
import torch
import numpy as np

device = 'cuda:0'
# choose the dataset and set the random seed
# the first run may be slow because the graph needs to be preprocessed into binary cache

np.random.seed(12306)
dataset = ["DBP_ZH_EN/", "DBP_JA_EN/", "DBP_FR_EN/", "SRPRS_FR_EN/", "SRPRS_DE_EN/", "DBP_WD/", "DBP_YG/"][2]
# dataset=  'EN_FR_15K_V1/'
# dataset= 'large_fr_strict/'
path = 'EA_datasets/' + dataset
# dataset= dataset+"EN"
# set hyper-parameters, load graphs and pre-aligned entity pairs
# if your GPU is out of memory, try to reduce the ent_dim
print("hello")

ent_dim, depth, top_k = 256, 2, 500
if "EN" in dataset:
    rel_dim, mini_dim = ent_dim // 2, 16
else:
    rel_dim, mini_dim = ent_dim // 3, 16

rel_dim = ent_dim

node_size, rel_size, ent_tuple2, triples_idx2, ent_ent2, ent_ent_val2, rel_ent2, ent_rel2 = load_graph(path)

train_pair, test_pair = load_aligned_pair(path, ratio=0.30)
candidates_x, candidates_y = set([x for x, y in test_pair]), set([y for x, y in test_pair])


# %%time


@torch.no_grad()
def my_norm(x):
    x /= (torch.norm(x, p=2, dim=-1).view(-1, 1) + 1e-8)
    return x


# main functions of LightEA
@torch.no_grad()
def get_random_vec(*dims, use_device=None):
    if use_device is None:
        use_device = device
    # print(dims)
    random_vec = torch.randn(*dims).to(use_device)
    random_vec = my_norm(random_vec)
    return random_vec


@torch.no_grad()
def random_projection(x, out_dim):
    random_vec = get_random_vec(x.shape[-1], out_dim, use_device=x.device)
    return x @ random_vec


@torch.no_grad()
def batch_sparse_matmul(sparse_tensor, dense_tensor, batch_size=32, save_mem=True):
    # batch_size =int(batch_size/8)
    if not isinstance(dense_tensor, torch.Tensor):
        dense_tensor = torch.from_numpy(dense_tensor).to(device)
    results = []
    for i in range(dense_tensor.shape[-1] // batch_size + 1):
        temp_result = torch.sparse.mm(sparse_tensor, dense_tensor[:, i * batch_size:(i + 1) * batch_size])
        if save_mem:
            temp_result = temp_result.cpu().numpy()
        results.append(temp_result)
    if save_mem:
        return np.concatenate(results, -1)
    else:
        return torch.cat(results, dim=-1)


@torch.no_grad()
def get_features(train_pair, extra_feature=None):
    train_pair = torch.tensor(train_pair).t()
    if extra_feature is not None:
        ent_feature = extra_feature
    else:
        random_vec = get_random_vec(train_pair.size(1), ent_dim)
        ent_feature = torch.zeros((node_size, ent_dim)).to(device)
        ent_feature[train_pair[0]] = random_vec
        ent_feature[train_pair[1]] = random_vec

    rel_feature = torch.zeros((rel_size, ent_feature.shape[-1])).to(device)
    ent_ent, ent_rel, rel_ent, ent_ent_val, triples_idx, ent_tuple = map(torch.tensor,
                                                                         [ent_ent2, ent_rel2, rel_ent2, ent_ent_val2,
                                                                          triples_idx2, ent_tuple2])
    list(map(lambda x: print(x.size()), [ent_ent, ent_rel, rel_ent, ent_ent_val, triples_idx, ent_tuple]))
    ent_ent, ent_rel, rel_ent, triples_idx, ent_tuple = map(lambda x: x.t(),
                                                            [ent_ent, ent_rel, rel_ent, triples_idx, ent_tuple])
    ent_ent_graph = torch.sparse_coo_tensor(indices=ent_ent, values=ent_ent_val, size=(node_size, node_size)).to(device)
    rel_ent_graph = torch.sparse_coo_tensor(indices=rel_ent, values=torch.ones(rel_ent.shape[1]),
                                            size=(rel_size, node_size)).to(device)
    ent_rel_graph = torch.sparse_coo_tensor(indices=ent_rel, values=torch.ones(ent_rel.shape[1]),
                                            size=(node_size, rel_size)).to(device)

    ent_list, rel_list = [ent_feature], [rel_feature]
    for i in trange(2):
        new_rel_feature = torch.from_numpy(batch_sparse_matmul(rel_ent_graph, ent_feature)).to(device)
        new_rel_feature = my_norm(new_rel_feature)

        new_ent_feature = torch.from_numpy(batch_sparse_matmul(ent_ent_graph, ent_feature)).to(device)
        new_ent_feature += torch.from_numpy(batch_sparse_matmul(ent_rel_graph, rel_feature)).to(device)
        new_ent_feature = my_norm(new_ent_feature)

        ent_feature = new_ent_feature;
        rel_feature = new_rel_feature
        ent_list.append(ent_feature);
        rel_list.append(rel_feature)

    ent_feature = torch.cat(ent_list, dim=1)
    rel_feature = torch.cat(rel_list, dim=1)

    ent_feature = my_norm(ent_feature)
    rel_feature = my_norm(rel_feature)
    rel_feature = random_projection(rel_feature, rel_dim)
    ent_feature = ent_feature.cpu()
    # = torch.tensor(triples_idx)
    batch_size = ent_feature.shape[-1] // mini_dim
    sparse_graph = torch.sparse_coo_tensor(indices=triples_idx, values=torch.ones(triples_idx.shape[1]),
                                           size=(torch.max(triples_idx).item() + 1, rel_size)).to(device)
    adj_value = batch_sparse_matmul(sparse_graph, rel_feature)
    del rel_feature
    # ent_featu

    features_list = []
    for batch in trange(rel_dim // batch_size + 1):
        temp_list = []
        for head in trange(batch_size):
            if batch * batch_size + head >= rel_dim:
                break
            sparse_graph = torch.sparse_coo_tensor(indices=ent_tuple, values=adj_value[:, batch * batch_size + head],
                                                   size=(node_size, node_size)).to(device)
            feature = batch_sparse_matmul(sparse_graph, random_projection(ent_feature, mini_dim).to(device),
                                          batch_size=128, save_mem=True)
            temp_list.append(feature)
        if len(temp_list):
            features_list.append(np.concatenate(temp_list, axis=-1))
            # features_list.append(K.concatenate(temp_list,-1).numpy())
    features = np.concatenate(features_list, axis=-1)

    faiss.normalize_L2(features)
    if extra_feature is not None:
        features = np.concatenate([ent_feature, features], axis=-1)
    return features


# obtain the literal features of entities, only work on DBP15K & SRPRS
# for the first run, you need to download the pre-train word embeddings from "http://nlp.stanford.edu/data/glove.6B.zip"
# unzip this file and put "glove.6B.300d.txt" into the root of LightEA

using_name_features = False
if using_name_features and "EN" in dataset:
    name_features = load_name_features(dataset, "./glove.6B.300d.txt", mode="hybrid-level")
    l_features = get_features(train_pair, extra_feature=name_features)


# %%time

# Obtain the structural features and iteratively generate Semi-supervised data
# "epoch = 1" represents removing the iterative strategy
#
# left,right = list(candidates_x),list(candidates_y)
# s_features = get_features(train_pair)
# import torch
# import pickle
# torch.save((left, right,s_features),path+'/trained_features.pkl',pickle_protocol=pickle.HIGHEST_PROTOCOL)


@torch.no_grad()
def sparse_sinkhorn_sims_pytorch(features_l, features_r, top_k=500, iteration=15, device='cuda:2', reg=0.02):
    faiss.normalize_L2(features_l)
    faiss.normalize_L2(features_r)
    print(features_l.shape, features_r.shape)

    dim, measure = features_l.shape[1], faiss.METRIC_INNER_PRODUCT
    param = 'Flat'
    index = faiss.index_factory(dim, param, measure)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 1, index)
    index.train(features_r)
    index.add(features_r)
    sims, index = index.search(features_l, top_k)
    sims = torch.tensor(sims).to(device)
    index = torch.tensor(index).to(device)
    # sims, index =(1-torch.cdist(torch.from_numpy(features_l).to(device),torch.from_numpy(features_r).to(device) )).topk(k=500, dim=-1)

    import torch_scatter
    #
    row_sims = torch.exp(sims.flatten() / reg)
    index = torch.flatten(index.to(torch.int64))

    size = features_l.shape[0]
    row_index = (torch.stack(
        [torch.arange(size * top_k).to(device) // top_k, index, torch.arange(size * top_k).to(device)])).t()
    col_index = row_index[torch.argsort(row_index[:, 1])]
    covert_idx = torch.argsort(col_index[:, 2])
    list(map(lambda x: print(x.size(), x.max(), x.min()), [row_sims, row_index, col_index, covert_idx, index]))

    for _ in range(iteration):
        row_sims = row_sims / torch_scatter.scatter_add(row_sims, row_index[:, 0])[row_index[:, 0]]
        col_sims = row_sims[col_index[:, 2]]
        col_sims = col_sims / torch_scatter.scatter_add(col_sims, col_index[:, 1])[col_index[:, 1]]
        row_sims = col_sims[covert_idx]

    index, sims = torch.reshape(row_index[:, 1], (-1, top_k)), torch.reshape(row_sims, (-1, top_k))
    return index, sims
    # return topk2spmat(sims, index, [features_l.shape[0], features_r.shape[0]])


def test_new(ranks, test_pair):
    left, right = test_pair[:, 0], np.unique(test_pair[:, 1])
    # index,sims = sparse_sinkhorn_sims(left, right,features,top_k,iteration,"test")
    # ranks = tf.argsort(-sims,-1).numpy()
    # index = index.numpy()

    wrong_list, right_list = [], []
    h1, h10, mrr = 0, 0, 0
    pos = np.zeros(np.max(right) + 1)
    pos[right] = np.arange(len(right))
    for i in range(len(test_pair)):
        rank = np.where(pos[test_pair[i, 1]] == index[i, ranks[i]])[0]
        if len(rank) != 0:
            if rank[0] == 0:
                h1 += 1
                right_list.append(test_pair[i])
            else:
                wrong_list.append((test_pair[i], right[index[i, ranks[i]][0]]))
            if rank[0] < 10:
                h10 += 1
            mrr += 1 / (rank[0] + 1)
    print("Hits@1: %.3f Hits@10: %.3f MRR: %.3f\n" % (h1 / len(test_pair), h10 / len(test_pair), mrr / len(test_pair)))

    return right_list, wrong_list


epochs = 3
for epoch in range(epochs):
    print("Round %d start:" % (epoch + 1))
    s_features = get_features(train_pair)
    if using_name_features and "EN" in dataset:
        features = np.concatenate([s_features, l_features], -1)
    else:
        features = s_features
    if epoch < epochs - 1:
        left, right = list(candidates_x), list(candidates_y)
        index, sims = sparse_sinkhorn_sims_pytorch(features[left], features[right], top_k)
        ranks = torch.argsort(-sims, -1).cpu().numpy()
        sims = sims.cpu().numpy();
        index = index.cpu().numpy()

        temp_pair = []
        x_list, y_list = list(candidates_x), list(candidates_y)
        for i in range(ranks.shape[0]):
            if sims[i, ranks[i, 0]] > 0.5:
                x = x_list[i]
                y = y_list[index[i, ranks[i, 0]]]
                temp_pair.append((x, y))

        for x, y in temp_pair:
            if x in candidates_x:
                candidates_x.remove(x);
            if y in candidates_y:
                candidates_y.remove(y);

        print("new generated pairs = %d" % (len(temp_pair)))
        print("rest pairs = %d" % (len(candidates_x)))

        if not len(temp_pair):
            break
        train_pair = np.concatenate([train_pair, np.array(temp_pair)])

    left, right = test_pair[:, 0], np.unique(test_pair[:, 1])
    torch.save((features[left], features[right]), path + f"/emb_{epoch}.pkl", pickle_protocol=pickle.HIGHEST_PROTOCOL)
    index, sims = sparse_sinkhorn_sims_pytorch(features[left], features[right], top_k)

    ranks = torch.argsort(-sims, -1).cpu().numpy()
    sims = sims.cpu().numpy();
    index = index.cpu().numpy()
    right_list, wrong_list = test_new(ranks, test_pair)