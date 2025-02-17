
# !git clone https://github.com/hyren/query2box.git


from ast import literal_eval
import pickle
import pandas as pd
import os.path
import numpy as np
import argparse
import itertools
import os
from tqdm import tqdm
from multiprocessing import Pool

def get_args():
    parser = argparse.ArgumentParser(description='Preprocess data for query2box')
    parser.add_argument('tag', type=str, choices=['train', 'test', 'valid'])
    parser.add_argument('--query_type', type=str, default='2c', help='Type of query to preprocess')
    parser.add_argument('--store_path', type=str, default='../data', help='Path to store preprocessed data')
    parser.add_argument('--dataset', type=str, default='FB15k', help='Dataset to preprocess')
    return parser.parse_args()

def read_data(path,dest='default.txt'):
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data

def get_raw_facts(query_type):
    if query_type == '1c':
        # raise ValueError('1c query type is already processed')
        path=f'../query2box/data/{args.dataset}/{args.tag}_ans_1c.pkl'
        dest='output2.txt'
        data=read_data(path,dest)
        flat_data = [{'head_entity_id': key[0][0], 
                    'relation_id': key[0][1][0], 
                    'tail_entity_ids': value} 
                    for key, value in data.items()]
        facts_1c_df = pd.DataFrame(flat_data)
        # facts_1c_df.head()
        return facts_1c_df

    elif query_type == '2c':
        path=f'../query2box/data/{args.dataset}/{args.tag}_ans_2c.pkl'
        dest='output2.txt'
        data=read_data(path,dest)
        flat_data = [{'head_entity_id': key[0][0], 
                    'relation_1_id': key[0][1][0], 
                    'relation_2_id': key[0][1][1], 
                    'tail_entity_ids': value} 
                    for key, value in data.items()]
        facts_2c_df = pd.DataFrame(flat_data)
        # facts_2c_df.head()
        return facts_2c_df

    elif query_type == '3c':
        path=f'../query2box/data/{args.dataset}/{args.tag}_ans_3c.pkl'
        dest='output2.txt'
        data=read_data(path,dest)
        flat_data = [{'head_entity_id': key[0][0], 
                    'relation_1_id': key[0][1][0], 
                    'relation_2_id': key[0][1][1],
                    'relation_3_id': key[0][1][2], 
                    'tail_entity_ids': value} 
                    for key, value in data.items()]
        facts_3c_df = pd.DataFrame(flat_data)
        # facts_3c_df.head()
        return facts_3c_df

    elif query_type == '2i':
        path=f'../query2box/data/{args.dataset}/{args.tag}_ans_2i.pkl'
        dest='output2.txt'
        data=read_data(path,dest)
        flat_data = [{'head_entity_1_id': key[0][0], 
                    'relation_1_id': key[0][1][0],
                    'head_entity_2_id': key[1][0], 
                    'relation_2_id': key[1][1][0], 
                    'tail_entity_ids': value} 
                    for key, value in data.items()]
        facts_2i_df = pd.DataFrame(flat_data)
        # facts_2i_df.head()
        return facts_2i_df
    
    elif query_type == '2u':
        path=f'../query2box/data/{args.dataset}/{args.tag}_ans_2u.pkl'
        dest='output2.txt'
        data=read_data(path,dest)
        flat_data = [{'head_entity_1_id': key[0][0], 
                    'relation_1_id': key[0][1][0],
                    'head_entity_2_id': key[1][0], 
                    'relation_2_id': key[1][1][0], 
                    'tail_entity_ids': value} 
                    for key, value in data.items()]
        facts_2u_df = pd.DataFrame(flat_data)
        return facts_2u_df

    elif query_type == '3i':
        path=f'../query2box/data/{args.dataset}/{args.tag}_ans_3i.pkl'
        dest='output2.txt'
        data=read_data(path,dest)
        flat_data = [{'head_entity_1_id': key[0][0], 
                    'relation_1_id': key[0][1][0],
                    'head_entity_2_id': key[1][0], 
                    'relation_2_id': key[1][1][0],
                    'head_entity_3_id': key[2][0],
                    'relation_3_id': key[2][1][0], 
                    'tail_entity_ids': value} 
                    for key, value in data.items()]
        facts_3i_df = pd.DataFrame(flat_data)
        # facts_3i_df.head()
        return facts_3i_df
    
    elif query_type in ["1i2u", "1u2i"]:
        facts_i2u = pd.read_csv(f'../query2box/data/{args.dataset}/{args.tag}_{query_type}.csv')
        facts_i2u['tail_entity_ids'] = facts_i2u['tail_entity_ids'].apply(literal_eval)
        return facts_i2u
    else:
        print('Invalid query type')
        return None

def add_subquery_columns(facts_df, query_type):
    def _get_tail_entities(facts_df, head_entity_column, relation_column, relation_idx):
        # get corresponding tail entities from data_ic for each subquery
        facts_df[f"tail_entity_{relation_idx}_ids"] = facts_df.apply(lambda x: 
                                    data_1c.loc[((x[head_entity_column], x[relation_column]))]['tail_entity_ids'] 
                                        if (x[head_entity_column], x[relation_column]) in data_1c.index
                                        else np.nan, axis=1)
        facts_df[f"tail_entity_{relation_idx}_intermediate_ids"] = facts_df.apply(lambda x:
                                    data_1c.loc[((x[head_entity_column], x[relation_column]))]['tail_entity_intermediate_ids']
                                        if (x[head_entity_column], x[relation_column]) in data_1c.index
                                            else np.nan, axis=1)
        facts_df[f"tail_entities_{relation_idx}"] = facts_df.apply(lambda x:
                                    data_1c.loc[((x[head_entity_column], x[relation_column]))]['tail_entities']
                                        if (x[head_entity_column], x[relation_column]) in data_1c.index
                                            else np.nan, axis=1)

        return facts_df

    head_entity_columns=[column for column in facts_df.columns if column.startswith('head_entity') and column.endswith('id') and "intermediate" not in column]
    relation_columns=[column for column in facts_df.columns if column.startswith('relation') and column.endswith('id')]

    
    if(query_type in ['2i','3i', '2u', '1i2u', '1u2i']):
        assert 'head_entity' not in  facts_df.columns
        # get appropriate column pairs
        for head_entity_column in tqdm(head_entity_columns, disable=True):
            for relation_column in relation_columns:
                if(head_entity_column != 'head_entity_id'):
                    head_entity_idx = head_entity_column.split('_')[2]
                    relation_idx = relation_column.split('_')[1]
                    
                    if(head_entity_idx == relation_idx):
                        facts_df = _get_tail_entities(facts_df, head_entity_column, relation_column, relation_idx)
                else:
                    relation_idx = relation_column.split('_')[1]
                    facts_df = _get_tail_entities(facts_df, head_entity_column, relation_column, relation_idx)
    elif(query_type in ['2c', '3c']):
        assert 'head_entity' in  facts_df.columns
        
        # rename head entity
        facts_df.rename(columns= {
            "head_entity" : "head_entity_1",
            "head_entity_id" : "head_entity_1_id",
            "head_entity_intermediate_id" : "head_entity_1_intermediate_id"
        }, inplace=True)


        relation_columns=[column for column in facts_df.columns if column.startswith('relation') and column.endswith('id')]
        num_relation = len(relation_columns)
        for i, relation in tqdm(enumerate(relation_columns), disable=True):
            relation_idx = int(relation.split('_')[1])
            facts_df = _get_tail_entities(facts_df, f"head_entity_{relation_idx}_id", relation, relation_idx)
            
            # combinatorial expansion
            if(i != num_relation-1):
                facts_df[f"head_entity_{relation_idx+1}"] = facts_df[f"tail_entities_{relation_idx}"]
                facts_df[f"head_entity_{relation_idx+1}_id"] = facts_df[f"tail_entity_{relation_idx}_ids"]
                facts_df[f"head_entity_{relation_idx+1}_intermediate_id"] = facts_df[f"tail_entity_{relation_idx}_intermediate_ids"]

                facts_df = facts_df.explode([f"head_entity_{relation_idx+1}", f"head_entity_{relation_idx+1}_id", f"head_entity_{relation_idx+1}_intermediate_id"], ignore_index=True)
                facts_df.dropna(inplace=True)
            
        # drop duplicates. We keep one tail entity out of all tail entities
        facts_df.drop_duplicates(subset=['head_entity_1_id'] + relation_columns, inplace=True)
    elif(query_type in ['1c']):
        return facts_df
    else:
        raise ValueError(f"query_type {query_type} not supported")
    facts_df.dropna(inplace=True)
    return facts_df



def good_name(facts_df, 
              index_to_entity_df, 
              index_to_relation_df,
              head_entity_columns=['head_entity_id'], 
              relation_columns=['relation_id']):
    # map entity
    for head_entity_column in tqdm(head_entity_columns, disable=True):
        facts_df = facts_df.merge(index_to_entity_df[['entity_id', 'entity_intermediate_id', 'entity', 'entity_type']], left_on=head_entity_column, right_on='entity_id')
        facts_df = facts_df.rename(columns=
                            {'entity': head_entity_column[:-3],
                             'entity_type': head_entity_column[:-3] + '_type',
                             'entity_intermediate_id': head_entity_column[:-3] + '_intermediate_id'
                            })
        facts_df = facts_df.drop(columns=['entity_id'])
    
    # map relation
    for relation in tqdm(relation_columns, disable=True):
        facts_df = facts_df.merge(index_to_relation_df, left_on=relation, right_on='relation_id')
        facts_df = facts_df.rename(columns={'relation': relation[:-3]})
        if(relation != 'relation_id'):
            facts_df = facts_df.drop(columns=['relation_id'])

    
    # map tail entities
    facts_df['tail_entities'] = facts_df['tail_entity_ids'].apply(lambda x: [index_to_entity_df[index_to_entity_df['entity_id'] == i]['entity'].item() for i in x])
    facts_df['tail_entity_intermediate_ids'] = facts_df['tail_entity_ids'].apply(lambda x: [index_to_entity_df[index_to_entity_df['entity_id'] == i]['entity_intermediate_id'].item() for i in x])

    # if single tail entity is none, remove the whole row
    # facts_df = facts_df[facts_df['tail_entities'].apply(lambda x: None not in x and len(x) == 1)]
    facts_df.dropna(inplace=True)
    return facts_df

if __name__ == '__main__':

    args = get_args()
    print(f"Starting data preprocessing: {args.dataset} {args.query_type} {args.tag}")

    if(args.query_type != '1c'):
        # already preprocessed 1c data. Combined all 1c data into one file
        data_1c = pd.DataFrame()
        for tag in ["train", "test", "valid"]:
            data_1c = pd.concat([data_1c, pd.read_csv(f'{args.store_path}/1c_data_preprocessed_{args.dataset}_{tag}.csv')])
        data_1c['tail_entities'] = data_1c['tail_entities'].apply(lambda x: literal_eval(x.replace('nan', 'None')))
        data_1c['tail_entity_ids'] = data_1c['tail_entity_ids'].apply(lambda x: literal_eval(x.replace('nan', 'None')))
        data_1c['tail_entity_intermediate_ids'] = data_1c['tail_entity_intermediate_ids'].apply(lambda x: literal_eval(x.replace('nan', 'None')))

        # resolve duplicate
        group_list = ['head_entity_id', 'relation_id']
        data_1c_grouped = []
        for key, df_group in tqdm(data_1c.groupby(group_list)):
            if(df_group.shape[0] > 1):
                df_group_resolved = {}
                for column in df_group.columns:
                    if(column.startswith("tail_ent")):
                        df_group_resolved[column] = list(dict.fromkeys(list(itertools.chain(*list(df_group[column].values)))))
                    else:
                        df_group_resolved[column] = df_group[column].iloc[0]
                data_1c_grouped.append(df_group_resolved)
            else:
                data_1c_grouped.append(df_group.iloc[0].to_dict())
        data_1c = pd.DataFrame(data_1c_grouped)
        # multi-index
        data_1c.set_index(['head_entity_id', 'relation_id'], inplace=True)
        # # find duplicate indices
        # duplicate_indices = data_1c.index.duplicated(keep='last')
        # if duplicate_indices.any():
        #     print(f"Duplicate indices found: {data_1c[duplicate_indices]}")
        #     data_1c = data_1c[~duplicate_indices]



    path=f'../query2box/data/{args.dataset}/ind2ent.pkl'
    dest='id_to_entity.txt'
    data=read_data(path,dest)
    index_to_entity_df = pd.DataFrame(list(data.items()), columns=['entity_id', 'entity_intermediate_id'])
    index_to_entity_df.head()


    path=f'../query2box/data/{args.dataset}/ind2rel.pkl'
    dest='id_to_relation.txt'
    data=read_data(path,dest)
    index_to_relation_df = pd.DataFrame(list(data.items()), columns=['relation_id', 'relation'])
    # index_to_relation_df.head()

    if not os.path.isfile(f'{args.store_path}/{args.dataset}_entity.csv'):
        entity_df_with_duplicate = pd.read_csv('../query2box/data/entity.txt', sep='\t', header=None)
        entity_df_with_duplicate.columns = ['entity_intermediate_id', 'entity', 'entity_type']
        entity_df_with_duplicate['entity_intermediate_id'] = entity_df_with_duplicate['entity_intermediate_id'].apply(lambda x: "/" + x.replace(".", "/"))
        entity_df = entity_df_with_duplicate[~entity_df_with_duplicate['entity_intermediate_id'].duplicated(keep=False)].sort_values(by='entity_intermediate_id')
        entity_df.to_csv(f'{args.store_path}/{args.dataset}_entity.csv', index=False)
    else:
        entity_df = pd.read_csv(f'{args.store_path}/{args.dataset}_entity.csv')
        entity_df.dropna(inplace=True)
        entity_df.index = entity_df['entity_intermediate_id']
        entity_df.drop('entity_intermediate_id', axis=1, inplace=True)
        # entity_df.head()


    # merge entity_df and index_to_entity_df
    index_to_entity_df = index_to_entity_df.merge(entity_df, left_on='entity_intermediate_id', right_on='entity_intermediate_id', how='left')
    # index_to_entity_df.head()

    store_filename = f'{args.store_path}/{args.query_type}_data_preprocessed_{args.dataset}_{args.tag}.csv'
    print(f"Removing {store_filename}")
    os.system(f"rm -f {store_filename}")
    def preprocess(fact_df):
        facts_df_good_name = good_name(fact_df, 
                                        index_to_entity_df, 
                                        index_to_relation_df,
                                        head_entity_columns=[column for column in fact_df.columns if column.startswith('head_entity')],
                                        relation_columns=[column for column in fact_df.columns if column.startswith('relation')])
        # print("Goodname mapping done")
        # print("Adding subquery columns")
        facts_df_good_name = add_subquery_columns(facts_df_good_name, args.query_type)
        # print("Subquery columns added")

        
        # check if file exists
        if(not os.path.isfile(store_filename)):
            facts_df_good_name.to_csv(store_filename, index=False)
        else:
            facts_df_good_name.to_csv(store_filename, index=False, mode='a', header=False)


    fact_df_all = get_raw_facts(args.query_type)


    num_pools = 16
    chunk_size = 1000
    total_chunks = fact_df_all.shape[0] // (chunk_size * num_pools) + 1

    print(f"chunk_size: {chunk_size * num_pools}")
    print(f"Total rows: {fact_df_all.shape[0]}")
    print(f"Total chunks: {total_chunks}")
    for i in tqdm(range(0, fact_df_all.shape[0], chunk_size * num_pools)):
        # print(f"Processing chunk {i//(chunk_size * num_pools)} / {total_chunks} and rows {i + chunk_size * num_pools}")
        pool = Pool(num_pools)
        pool.map(preprocess, np.array_split(fact_df_all[i : i + chunk_size * num_pools], num_pools))
        pool.close()
        pool.join()
            
    print("Done")
    del fact_df_all
    if(args.query_type != '1c'):
        del data_1c
    del index_to_entity_df
    del index_to_relation_df
    del entity_df
    del pool
    



