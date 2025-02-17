
from ogb.lsc import WikiKG90Mv2Dataset
dataset = WikiKG90Mv2Dataset(root="../query2box/data/wiki/")


print(dataset.num_entities) # number of entities
print(dataset.num_relations) # number of relation types
print(dataset.num_feat_dims) # dimensionality of entity/relation features.


print(dataset.relation_feat.shape)
print(dataset.all_relation_feat.shape)


import pandas as pd
entity_df = pd.read_csv("../query2box/data/wiki/wikikg90mv2_mapping/entity.csv")
entity_df.index = entity_df.idx
entity_df.shape


entity_df





import pickle
ind2ent = dict(zip(entity_df['idx'], entity_df['idx']))
with open("../query2box/data/wiki/ind2ent.pkl", "wb") as f:
    pickle.dump(ind2ent, f)


entity_df_store = pd.DataFrame()
entity_df_store['entity_intermediate_id'] = entity_df['idx']
entity_df_store['entity'] = entity_df['title']
entity_df_store['entity_type'] = "dummy"

entity_df_store.to_csv("../data_optimized/wiki_entity.csv", index=False)
entity_df_store


relation_df = pd.read_csv("../query2box/data/wiki/wikikg90mv2_mapping/relation.csv")
relation_df.index = relation_df.idx
relation_df.shape


relation_df


ind2rel = dict(zip(relation_df['idx'], relation_df['title']))
with open("../query2box/data/wiki/ind2rel.pkl", "wb") as f:
    pickle.dump(ind2rel, f)


train_hrt = dataset.train_hrt # numpy ndarray of shape (num_triples, 3)
print(train_hrt[0:5]) # get i-th training triple (h[i], r[i], t[i])



max_len = 10000000
index_array = list(range(0, max_len))
import random
random.seed(0)
random.shuffle(index_array)


# split index_array into train, test and valid index with ratio 50:25:25
train_len = max_len // 2
train_indices = index_array[:train_len]
valid_indices = index_array[train_len: (train_len * 3)//2]
test_indices = index_array[(train_len * 3)//2:max_len]

# print(train_indices)
# print(valid_indices)
# print(test_indices)

from tqdm import tqdm
def prepare(indices):
    ans_1c = {}
    csv_1c = {
        "head_entity_id": [],
        "relation_id": [],
        "tail_entity_ids": [],
        "head_entity_intermediate_id": [],
        "head_entity": [],
        "head_entity_type": [],
        "relation": [],
        "tail_entities": [],
        "tail_entity_intermediate_ids": []
    }

    for i in tqdm(indices):
        h, r, t = train_hrt[i]
        ans_1c[((h, (r, )),)] = {t}
        csv_1c["head_entity_id"].append(h)
        csv_1c["relation_id"].append(r)
        csv_1c["tail_entity_ids"].append({t})
        csv_1c["head_entity_intermediate_id"].append(entity_df.iloc[h]['idx'])
        csv_1c["head_entity"].append(entity_df.iloc[h]['title'])
        csv_1c["head_entity_type"].append("dummy")
        csv_1c["relation"].append(relation_df.iloc[r]['title'])
        csv_1c["tail_entities"].append([entity_df.iloc[t]['title']])
        csv_1c["tail_entity_intermediate_ids"].append([entity_df.iloc[t]['idx']])
        
    return ans_1c, csv_1c


train_ans_1c, train_csv_1c = prepare(train_indices)
valid_ans_1c, valid_csv_1c = prepare(valid_indices)
test_ans_1c, test_csv_1c = prepare(test_indices)


train_1c_df = pd.DataFrame(train_csv_1c)
train_1c_df.to_csv("../data_optimized/1c_data_preprocessed_wiki_train.csv", index=False)
with open("../query2box/data/wiki/train_ans_1c.pkl", "wb") as f:
    pickle.dump(train_ans_1c, f)

valid_1c_df = pd.DataFrame(valid_csv_1c)
valid_1c_df.to_csv("../data_optimized/1c_data_preprocessed_wiki_valid.csv", index=False)
with open("../query2box/data/wiki/valid_ans_1c.pkl", "wb") as f:
    pickle.dump(valid_ans_1c, f)

test_1c_df = pd.DataFrame(test_csv_1c)
test_1c_df.to_csv("../data_optimized/1c_data_preprocessed_wiki_test.csv", index=False)
with open("../query2box/data/wiki/test_ans_1c.pkl", "wb") as f:
    pickle.dump(test_ans_1c, f)