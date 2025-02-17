import pickle
import pandas as pd
import argparse

def read_data(path,dest='default.txt'):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def get_args():
    parser = argparse.ArgumentParser(description='Preprocess data for query2box')
    parser.add_argument('tag', type=str, choices=['test', 'valid'])
    parser.add_argument('query_type', type=str, choices=['1i2u', '1u2i'])
    parser.add_argument('--dataset', type=str, default='FB15k', help='Dataset to preprocess')
    return parser.parse_args()


if __name__ == '__main__':


    """
        New code
    """

    args = get_args()

    
    if(args.query_type == '1i2u'):
        # 2i
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
        print(facts_2i_df.shape)
        facts_2i_df.head()


        facts_1i2u_df = pd.DataFrame()
        for i in range(facts_2i_df.shape[0]):
            mask = (facts_2i_df['head_entity_1_id'] ==  facts_2i_df['head_entity_1_id'].iloc[i]) \
                    & (facts_2i_df['relation_1_id'] == facts_2i_df['relation_1_id'].iloc[i]) \
                    & (facts_2i_df.index != i)
            facts_2i_df_index = facts_2i_df[mask].copy()
            if facts_2i_df_index.shape[0] > 0:
                facts_2i_df_index = facts_2i_df_index.iloc[:1]
                facts_2i_df_index['head_entity_3_id'] = facts_2i_df['head_entity_2_id'].iloc[i]
                facts_2i_df_index['relation_3_id'] = facts_2i_df['relation_2_id'].iloc[i]
                # facts_2i_df_index['tail_entity_prev_len'] = facts_2i_df_index['tail_entity_ids'].apply(lambda x: len(x))
                facts_2i_df_index['tail_entity_ids'] = facts_2i_df_index['tail_entity_ids'].apply(lambda x: x.union(facts_2i_df['tail_entity_ids'].iloc[i]))
                facts_1i2u_df = pd.concat([facts_1i2u_df, facts_2i_df_index], ignore_index=True)
        
        # save
        print(facts_1i2u_df.shape)
        facts_1i2u_df.to_csv(f'../query2box/data/{args.dataset}/{args.tag}_1i2u.csv', index=False)


    elif(args.query_type == '1u2i'):
        # 2u
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
        print(facts_2u_df.shape)
        facts_2u_df.head()



        facts_1u2i_df = pd.DataFrame()
        for i in range(facts_2u_df.shape[0]):
            # check first position
            mask = (facts_2u_df['head_entity_1_id'] ==  facts_2u_df['head_entity_1_id'].iloc[i]) \
                    & (facts_2u_df['relation_1_id'] == facts_2u_df['relation_1_id'].iloc[i]) \
                    & (facts_2u_df.index != i)
            facts_2u_df_index = facts_2u_df[mask].copy()
            if facts_2u_df_index.shape[0] > 0:
                facts_2u_df_index = facts_2u_df_index.iloc[:1]
                facts_2u_df_index['head_entity_3_id'] = facts_2u_df['head_entity_2_id'].iloc[i]
                facts_2u_df_index['relation_3_id'] = facts_2u_df['relation_2_id'].iloc[i]
                # facts_2u_df_index['tail_entity_prev_len'] = facts_2u_df_index['tail_entity_ids'].apply(lambda x: len(x))
                facts_2u_df_index['tail_entity_ids'] = facts_2u_df_index['tail_entity_ids'].apply(lambda x: x.intersection(facts_2u_df['tail_entity_ids'].iloc[i]))
                facts_1u2i_df = pd.concat([facts_1u2i_df, facts_2u_df_index], ignore_index=True)
        
            # check second position
            mask = (facts_2u_df['head_entity_2_id'] ==  facts_2u_df['head_entity_1_id'].iloc[i]) \
                    & (facts_2u_df['relation_2_id'] == facts_2u_df['relation_1_id'].iloc[i]) \
                    & (facts_2u_df.index != i)
            facts_2u_df_index = facts_2u_df[mask].copy()
            if facts_2u_df_index.shape[0] > 0:
                facts_2u_df_index = facts_2u_df_index.iloc[:1]

                # swap column names
                facts_2u_df_index[['head_entity_1_id', 'relation_1_id', 'head_entity_2_id', 'relation_2_id']] = \
                    facts_2u_df_index[['head_entity_2_id', 'relation_2_id', 'head_entity_1_id', 'relation_1_id']]
                
                facts_2u_df_index['head_entity_3_id'] = facts_2u_df['head_entity_2_id'].iloc[i]
                facts_2u_df_index['relation_3_id'] = facts_2u_df['relation_2_id'].iloc[i]
                # facts_2u_df_index['tail_entity_prev_len'] = facts_2u_df_index['tail_entity_ids'].apply(lambda x: len(x))
                facts_2u_df_index['tail_entity_ids'] = facts_2u_df_index['tail_entity_ids'].apply(lambda x: x.intersection(facts_2u_df['tail_entity_ids'].iloc[i]))
                facts_1u2i_df = pd.concat([facts_1u2i_df, facts_2u_df_index], ignore_index=True)
        
        # save
        print(facts_1u2i_df.shape)
        facts_1u2i_df.to_csv(f'../query2box/data/{args.dataset}/{args.tag}_1u2i.csv', index=False)