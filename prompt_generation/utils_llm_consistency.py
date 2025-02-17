import pandas as pd
import numpy as np
import textwrap

def get_context(subgraph, 
                target_head_entities, 
                target_relations,
                target_tail_entities,
                relation_separator,
                tokenizer = None,
                max_context_len = 1000,
                verbose=False):

    
    assert len(target_head_entities) == len(target_relations), "Number of target_head_entities and target_relations should be the same"
    
    subgraph = pd.DataFrame(subgraph, columns=['head_entity', 'relation', 'tail_entity'])
    subgraph.dropna(inplace=True)
    context_full = subgraph.copy()
    context = pd.DataFrame()

    global avg_triplet_token_length, count_triplet
    avg_triplet_token_length = 0
    count_triplet = 0
    def get_token_length(context, verbose=False):
        context_as_string = ("\n").join([(" | ").join([head_entity_context, relation_context.split(relation_separator)[-1], tail_entity_context])
            for head_entity_context, relation_context, tail_entity_context in context.values]).replace("_", " ")
        
        
        token_length = len(tokenizer.encode(context_as_string))
        global avg_triplet_token_length, count_triplet
        if(context.shape[0] != 0):
            if(count_triplet == 0):
                # initialize
                avg_triplet_token_length = token_length / context.shape[0]
            else:
                # update
                avg_triplet_token_length = (avg_triplet_token_length * count_triplet + token_length) / (count_triplet + context.shape[0])
        count_triplet += context.shape[0]
        if(verbose):
            print(f"Average token length per triplet: {avg_triplet_token_length}")
            print(f"Total token length: {token_length}")
            print(f"Total triplets: {count_triplet}")
        return  token_length
            
   


    for target_head_entity, target_relation in zip(target_head_entities, target_relations):
        max_context_len_per_relation = max_context_len // len(target_relations) # divide the max_context_len by the number of target_relations
        if(verbose):
            print()
            print(target_head_entity, target_relation)

        if(relation_separator == "/"):
            relation_split = target_relation.split("/")
        elif(relation_separator == ":"):
            # split the relation string in 4 equal parts
            chunk = 4
            # relation_split = list(map(''.join, zip(*[iter(target_relation)]*(len(target_relation)//chunk))))
            relation_split = textwrap.wrap(target_relation, len(target_relation)//chunk)
        elif(relation_separator == " "):
            relation_split = target_relation.split(" ")
        else:
            raise ValueError(f"{relation_separator} is not recognized")   
    
        # add empty split
        relation_split = [""] + relation_split
        if(verbose):
            print(f"Relation split: {relation_split}")
            print(f"Subgraph dim: {subgraph.shape}")
        
        # start with the most specific relation and then go along the hierarchy
        for i in range(len(relation_split), -1, -1):
            if(relation_separator == "/"):
                relation_substring = ("/").join(relation_split[:i+1])
            elif(relation_separator == ":"):
                relation_substring = ("").join(relation_split[:i+1])
            elif(relation_separator == " "):
                relation_substring = (" ").join(relation_split[:i+1])
            else:
                raise ValueError(f"{relation_separator} is not recognized")
            
            if(i == len(relation_split)):
                # most specific triplets
                mask = (context_full['head_entity'] == target_head_entity) \
                        & (context_full['relation'] == target_relation) \
                        & (context_full['tail_entity'].isin(target_tail_entities))
            else:
                mask = context_full['relation'].str.startswith(relation_substring)
            
            
            context_masked = context_full[mask]
            additional_df_to_be_concatenated = None


            token_length_context = get_token_length(context_masked, verbose=verbose)
            if(token_length_context > max_context_len_per_relation):
                chosen_idx_within_mask = np.random.choice(context_masked.shape[0], 
                                                          replace=False, 
                                                          size=min(context_masked.shape[0], int(max_context_len_per_relation // avg_triplet_token_length)))
                additional_df_to_be_concatenated = context_masked.iloc[np.array(list(set(np.arange(context_masked.shape[0])) - set(chosen_idx_within_mask)))]
                context_masked = context_masked.iloc[chosen_idx_within_mask] 
            
            context = pd.concat([context, context_masked])
            context_full = context_full[~mask] # remaining context
            if(additional_df_to_be_concatenated is not None):
                context_full = pd.concat([context_full, additional_df_to_be_concatenated])
            max_context_len_per_relation -= token_length_context

            if(verbose):
                print(f"Relation starts as {relation_substring} -- found context length {context.shape[0]}")
            if(max_context_len_per_relation <= 0):
                break

    # a random shuffle of the context
    context = context.sample(frac=1).reset_index(drop=True)
    return context


def prune(df_row, filename, max_tail_entities=None):
    """
        1. Remove None values from the flipped_tail_entities and tail_entities columns. Apply for subqueries
        2. Keep only the first max_tail_entities. For subqueries, keep first 2 * max_tail_entities
    """
    df_row = df_row.copy()

    assert df_row.shape[0] == 1, "Input should be a single row dataframe"
    num_subqueries = 0
    # Remove None tail entities
    for column in df_row.columns:
        if("flipped_tail_entities" in column):
            tail_entity_column = column.replace("flipped_", "")
            assert tail_entity_column in df_row.columns, "tail_entity_column should be in the dataframe"
            
            flipped_entities_keep = []
            entities_keep = []
            for i, entity in enumerate(df_row.iloc[0][tail_entity_column]):
                if(entity is not None):
                    entities_keep.append(entity)
                    assert df_row.iloc[0][tail_entity_column][i] is not None, "tail_entity should not be None"
                    flipped_entities_keep.append(df_row.iloc[0][column][i])
            
            df_row[column] = [flipped_entities_keep]
            df_row[tail_entity_column] = [entities_keep]

            if(column != "flipped_tail_entities"):
                num_subqueries += 1


    
    if(max_tail_entities != None):

        subgraph = pd.DataFrame(df_row.iloc[0]['subgraph'], columns=['head_entity', 'relation', 'tail_entity'])
        if("i_data_final" in filename or 
           "u_data_final" in filename or 
           "1u2i_data_final" in filename or 
           "1i2u_data_final" in filename
        ):
            
            
            tail_entites_deleted = []
            for i in range(num_subqueries):
                
                tail_entites_deleted.append(df_row.iloc[0]['tail_entities_{}'.format(i+1)][max_tail_entities:])
                df_row['tail_entities_{}'.format(i+1)] = [df_row.iloc[0]['tail_entities_{}'.format(i+1)][:max_tail_entities]]
                df_row['flipped_tail_entities_{}'.format(i+1)] = [df_row.iloc[0]['flipped_tail_entities_{}'.format(i+1)][:max_tail_entities]]

            
            
            for i in range(num_subqueries):
                mask = (subgraph['tail_entity'].isin(tail_entites_deleted[i])) & \
                    (subgraph['head_entity'] == df_row.iloc[0][f'head_entity_{i+1}']) & \
                    (subgraph['relation'] == df_row.iloc[0][f'relation_{i+1}'])
                subgraph = subgraph[~mask]
            

        
        else:

            # single query
            tail_entities_delete = df_row.iloc[0]['tail_entities'][max_tail_entities:]
            tail_entities_keep = df_row.iloc[0]['tail_entities'][:max_tail_entities]
            flipped_tail_entiities_keep = df_row.iloc[0]['flipped_tail_entities'][:max_tail_entities]
            df_row['tail_entities'] = [tail_entities_keep]
            df_row['flipped_tail_entities'] = [flipped_tail_entiities_keep]
        

            # update subgraph
            mask = subgraph['tail_entity'].isin(tail_entities_delete) & \
                (subgraph['head_entity'] == df_row.iloc[0]['head_entity']) & \
                (subgraph['relation'] == df_row.iloc[0]['relation'])
            subgraph = subgraph[~mask]

        
        df_row['subgraph'] = [list(zip(subgraph['head_entity'], subgraph['relation'], subgraph['tail_entity']))]
            

    return df_row