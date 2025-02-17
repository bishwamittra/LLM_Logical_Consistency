
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import networkx as nx
import os
import numpy as np
import argparse
from time import time
from multiprocessing import Pool

def get_args():
    parser = argparse.ArgumentParser(description='Preprocess data for query2box')
    parser.add_argument('filename', type=str, default='1c_data_preprocessed_FB15k_test.csv', help='Filename to generate data for')
    parser.add_argument('--store_path', type=str, default='../data', help='Path to store preprocessed data')
    parser.add_argument("--bfs_depth", type=int, default=2, help="BFS depth for subgraph")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to process")
    parser.add_argument('--dataset', type=str, default='FB15k', help='Dataset to preprocess')
    return parser.parse_args()

args = get_args()

dataset = args.filename.split('_')[-2]
# ### Create a multi directed graph from a single chain dataset
data_1c = pd.DataFrame()
for tag in ["train", "test", "valid"]:
    data_1c = pd.concat([data_1c, pd.read_csv(f'{args.store_path}/1c_data_preprocessed_{dataset}_{tag}.csv')])
    
data_1c.dropna(inplace=True)
data_1c['tail_entities'] = data_1c['tail_entities'].apply(lambda x: literal_eval(x.replace('nan', 'None')))
data_1c['tail_entity_ids'] = data_1c['tail_entity_ids'].apply(lambda x: literal_eval(x.replace('nan', 'None')))
data_1c['tail_entity_intermediate_ids'] = data_1c['tail_entity_intermediate_ids'].apply(lambda x: literal_eval(x.replace('nan', 'None')))
data_1c = data_1c[['head_entity_intermediate_id', 'tail_entity_intermediate_ids', 'relation']]
data_1c_explode = data_1c.explode(['tail_entity_intermediate_ids'])
data_1c_explode = data_1c_explode.rename(columns={
        'tail_entity_intermediate_ids': 'tail_entity_intermediate_id'
})
data_1c_explode.dropna(inplace=True)
relation_column_name = 'relation'
graph = nx.from_pandas_edgelist(data_1c_explode, 'head_entity_intermediate_id', 'tail_entity_intermediate_id', [relation_column_name], create_using=nx.MultiDiGraph())


# ### Helper functions and variables
# pre-processing
entity_df = pd.read_csv(f'{args.store_path}/{args.dataset}_entity.csv')
entity_df.dropna(inplace=True)
entity_df.index = entity_df['entity_intermediate_id']
entity_df.drop('entity_intermediate_id', axis=1, inplace=True)
# prune entity_df where entity starts with common
entity_df = entity_df[~entity_df['entity'].str.startswith("common.")]
entity_df_index_dict = set(entity_df.index)


if(not os.path.exists(f'{args.store_path}/entity_type_to_intermediate_id_{dataset}.csv')):
    entity_type_to_intermediate_id = {}
    for i, row in entity_df.iterrows():
        if row['entity_type'] not in entity_type_to_intermediate_id:
            entity_type_to_intermediate_id[row['entity_type']] = []
        entity_type_to_intermediate_id[row['entity_type']].append(i)

    entity_type_to_intermediate_id_df = pd.DataFrame([entity_type_to_intermediate_id]).transpose()
    entity_type_to_intermediate_id_df.columns = ['entity_intermediate_ids']
    entity_type_to_intermediate_id_df['entity_type'] = entity_type_to_intermediate_id_df.index
    entity_type_to_intermediate_id_df.reset_index(drop=True, inplace=True)

    # store entity_type_to_intermediate_id_df
    entity_type_to_intermediate_id_df.to_csv(f'{args.store_path}/entity_type_to_intermediate_id_{dataset}.csv', index=False)
    entity_type_to_intermediate_id_df.index = entity_type_to_intermediate_id_df['entity_type']
    entity_type_to_intermediate_id_df.drop('entity_type', axis=1, inplace=True)

else:
    entity_type_to_intermediate_id_df = pd.read_csv(f'{args.store_path}/entity_type_to_intermediate_id_{dataset}.csv')
    entity_type_to_intermediate_id_df['entity_intermediate_ids'] = entity_type_to_intermediate_id_df['entity_intermediate_ids'].apply(lambda x: literal_eval(x))
    entity_type_to_intermediate_id_df.index = entity_type_to_intermediate_id_df['entity_type']
    entity_type_to_intermediate_id_df.drop('entity_type', axis=1, inplace=True)



def get_flipped_answer(tail_entity_intermediate_ids):
    result = None
    for tail_entity_intermediate_id in tail_entity_intermediate_ids:
        if(tail_entity_intermediate_id not in entity_df_index_dict):
            continue
        entity_type = entity_df.loc[tail_entity_intermediate_id]['entity_type']


        # get a random entity of the same type and check if it is not in the tail_entity_intermediate_ids
        flipped_entity_intermediate_id = None
        i = 0
        success = False
        while(True):
            i += 1
            flipped_entity_intermediate_id = np.random.choice(entity_type_to_intermediate_id_df.loc[entity_type]['entity_intermediate_ids'])
            if flipped_entity_intermediate_id not in  tail_entity_intermediate_ids:
                success = True
                break
            if i > 100:
                break
        if success:
            # result.append(entity_df.loc[flipped_entity_intermediate_id]['entity'])
            result = entity_df.loc[flipped_entity_intermediate_id]['entity']
            if(result.startswith("common.")):
                continue
            else:
                break
    # print([result] * len(tail_entity_intermediate_ids))
    # quit()
    return [result] * len(tail_entity_intermediate_ids)


def bounded_bfs(start_node, max_depth=1):
        depth = 0
        visited = {}
        queue = []
        queue.append(start_node)
        visited[start_node] = True
        selected_edges = []
        while queue:
            depth_element_size = len(queue)
            while depth_element_size > 0:
                s = queue.pop(0)
                for i in graph.neighbors(s): #TODO: use a dict to store the edges
                    if i not in visited:
                        queue.append(i)
                        visited[i] = True

                        edges_info = graph.get_edge_data(s, i)
                        if edges_info:
                            for edge_key, edge_data in edges_info.items():
                                edge_name = edge_data.get(relation_column_name, None)
                                selected_edges.append((s, i, edge_name))
                                
                depth_element_size -= 1
            depth += 1
            if(depth >= max_depth):
                break
        return selected_edges



def get_subgraph(start_node, max_depth=1, relation_column_name='relation', verbose=False):
    start_time = time()
    selected_edges = bounded_bfs(start_node, max_depth=max_depth)
    time_bfs = time() - start_time
    if(verbose):
        print("Selected Edges:", len(selected_edges))

    start_time = time()
    edge_list_good_name = []
    for edge in selected_edges:
        try:
            edge_list_good_name.append((entity_df.loc[edge[0]]['entity'], edge[2], entity_df.loc[edge[1]]['entity']))
        except:
            pass
    if(verbose):
        print("Selected Edges with goodname:", len(edge_list_good_name))
    time_good_name = time() - start_time
    
    return edge_list_good_name, time_bfs, time_good_name

def generate_data(data):
    

    data.dropna(inplace=True)
    for column in data.columns:
        if(column.startswith('tail')):
            data[column] = data[column].apply(lambda x: literal_eval(x.replace('nan', 'None')))

    # for later use
    head_entity_columns = [column for column in data.columns if column.startswith('head_entity') and column.endswith('intermediate_id')]
    tail_entity_columns = [column for column in data.columns if column.startswith('tail_entit') and not column.endswith('_ids')]

    # columns to store
    columns_to_store = []
    for column in data.columns:
        if(column.startswith('head_entity') and not column.endswith('id') and not column.endswith('type')):
            columns_to_store.append(column)
        elif(column.startswith('relation') and not column.endswith('id')):
            columns_to_store.append(column)
        elif(column.startswith('tail_entit') and not column.endswith('ids')):
            columns_to_store.append(column)
            columns_to_store.append(f"flipped_{column}")
    columns_to_store.append('subgraph')
    columns_to_store

    # max_index = len(data)
    max_index = data.shape[0]
    df_result = pd.DataFrame()
    for i in tqdm(range(max_index)):
        row = data.iloc[i].copy()

        head_entity_subgraph_dict = {}
        time_subgraph = 0
        time_bfs = 0
        time_bfs_good_name = 0
        non_cached_entities = 0
        for head_entity_column in head_entity_columns:
            head_entity_idx = "" if head_entity_column == 'head_entity_intermediate_id' else head_entity_column.split('_')[2]
            entity_intermediate_id = row[f"head_entity_{head_entity_idx}_intermediate_id"] if head_entity_idx != "" else row['head_entity_intermediate_id']
                        
            non_cached_entities += 1
            start_time = time()
            entity_subgraph, time_bfs_per_head, time_bfs_good_name_per_head = get_subgraph(entity_intermediate_id, max_depth=args.bfs_depth, relation_column_name='relation', verbose=False)
            time_subgraph += time() - start_time
            time_bfs += time_bfs_per_head
            time_bfs_good_name += time_bfs_good_name_per_head
            
            head_entity_subgraph_dict[head_entity_column] = list(set(entity_subgraph))

        combined_subgraph = []
        for key in head_entity_subgraph_dict:
            combined_subgraph += head_entity_subgraph_dict[key]    


        row['subgraph'] = list(set(combined_subgraph))
        if(non_cached_entities > 0):
            row['time_subgraph'] = time_subgraph
            row['time_bfs'] = time_bfs
            row['time_bfs_good_name'] = time_bfs_good_name
        else:
            row['time_subgraph'] = None
            row['time_bfs'] = None
            row['time_bfs_good_name'] = None

    
        if(True):
            # get flipped answer for each tail_entity_intermediate_ids
            for tail_entity_column in tail_entity_columns:
                start_time = time()
                tail_entity_idx = "" if tail_entity_column.endswith("entities") else tail_entity_column.split('_')[-1]
                tail_entity_intermediate_ids = row[f"tail_entity_{tail_entity_idx}_intermediate_ids"] if tail_entity_idx != "" else row['tail_entity_intermediate_ids']
                flipped_tail_entities = get_flipped_answer(tail_entity_intermediate_ids)
                row[f'flipped_tail_entities_{tail_entity_idx}' if tail_entity_idx != "" else "flipped_tail_entities"] = flipped_tail_entities
                row[f'time_flipped_entity_find_{tail_entity_idx}' if tail_entity_idx != "" else "time_flipped_entity_find"] = (time() - start_time)

        # store as a pandas dataframe
        df_temp = pd.DataFrame([row[columns_to_store + [column for column in row.index.tolist() if column.startswith('time')]]])
        df_result = pd.concat([df_result, df_temp], ignore_index=True)

    # check if file exists
    if(not os.path.exists(store_file_name)):
        df_result.to_csv(store_file_name, index=False)
    else:
        df_result.to_csv(store_file_name, mode='a', header=False, index=False)

# ### Prepare dataset on single chain
filename = args.filename
total_lines = sum(1 for row in open(filename, 'r'))
print(f"Total lines in {filename}: {total_lines}")
chunksize = 10
chunkidx = 0
total_chunks = total_lines//chunksize + 1

# store file_name
store_file_name = f'{filename.split("_data_preprocessed")[0]}_data_final_{filename.split("_")[-2]}_{filename.split("_")[-1].split(".")[0]}_{args.bfs_depth}.csv'
print(f"Store file name: {store_file_name}")
os.system(f"rm -f {store_file_name}")

num_pools = 16
print(f"Chunksize: {chunksize}")
print(f"Total chunks: {total_chunks}")
with pd.read_csv(filename, chunksize=chunksize, nrows=args.nrows) as reader:
    data_pool = []
    for data in reader:
        data_pool.append(data)
        chunkidx += 1
        if(chunkidx % num_pools == 0 or chunkidx == total_chunks):
            print(f"Loaded chunk {chunkidx}/{total_chunks} from {filename} with shape {data.shape}")
            pool = Pool(num_pools)
            pool.map(generate_data, data_pool)
            pool.close()
            pool.join()
            data_pool = []
        
        
    # free memory
    del graph
    del entity_df
    del entity_df_index_dict
    del entity_type_to_intermediate_id_df
    del data_1c
    del pool



        