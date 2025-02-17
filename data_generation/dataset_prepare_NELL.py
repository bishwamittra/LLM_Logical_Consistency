
# !git clone https://github.com/hyren/query2box.git


from ast import literal_eval
import pickle
import pandas as pd
import os.path
import numpy as np

def read_data(path,dest='default.txt'):
  with open(path, 'rb') as file:
    data = pickle.load(file)
  return data
 

path='../query2box/data/NELL/ind2ent.pkl'
dest='id_to_entity.txt'
data=read_data(path,dest)
index_to_entity_df = pd.DataFrame(list(data.items()), columns=['entity_id', 'entity_intermediate_id'])
print(index_to_entity_df.shape)
index_to_entity_df.head(n=10)




index_to_entity_df = index_to_entity_df[index_to_entity_df["entity_intermediate_id"].str.startswith("concept")]
index_to_entity_df



entity_df = pd.DataFrame()
entity_df['entity_intermediate_id'] = index_to_entity_df['entity_intermediate_id']
entity_df['entity'] = index_to_entity_df['entity_intermediate_id'].apply(lambda x: ("_").join(x.split('_')[2:]))
entity_df['entity_type'] = index_to_entity_df['entity_intermediate_id'].apply(lambda x: x.split('_')[1])
entity_df


entity_df.to_csv("../data_optimized/NELL_entity.csv", index=False)



