
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import networkx as nx
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
import json
from time import time
import numpy as np
from utils_llm_consistency import get_context, prune
import argparse
from utils import get_logger
from vllm import LLM, SamplingParams
from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--model_name", type=str, default="llama7b_chat_hf", choices=["llama13b_hf", "llama13b_chat_hf", "llama7b_hf", "llama7b_chat_hf"])
parser.add_argument("--context_length", type=int, default=1000)
parser.add_argument("--nrows", type=int, default=None)
# parser.add_argument("--cuda_id", type=int, default=1)
args = parser.parse_args()


save_root = "result"
logger, exp_seq, save_path = get_logger(save_root=save_root, save_tag="negation_consistency")
logger.info(f"=======Exp: {exp_seq}=============")
logger.info(f"Model: {args.model_name}")





config = None
with open("../config.json") as f:
    config = json.load(f)
if(config == None):
    quit()

llm = LLM(model=config[args.model_name]['model_path'])
sampling_params = SamplingParams(temperature=config[args.model_name]['temperature'], 
                                 max_tokens=config[args.model_name]['max_tokens'],
                                 top_k=config[args.model_name]['top_k'])





filename = args.filename
assert "data_final" in filename
assert "/" in filename
store_filename = f"{filename.split('/')[-1].replace('data_final', 'prompt')}"
assert "/" not in store_filename
logger.info(f"Store filename: {store_filename}")
total_lines = sum(1 for row in open(filename, 'r'))
logger.info(f"Total lines in {filename}: {total_lines}")
chunksize = 100
chunkidx = 0
total_chunks = total_lines//chunksize + 1
logger.info(f"Chunksize: {chunksize}")
logger.info(f"Total chunks: {total_chunks}")
with pd.read_csv(filename, chunksize=chunksize, nrows=args.nrows) as reader:
    for data in reader:
        logger.info(f"Loaded chunk {chunkidx + 1}/{total_chunks} from {filename} with shape {data.shape}")
        chunkidx += 1
        # convert to list
        for column in data.columns:
            if("subgraph" in column or "tail_entities" in column) and "time" not in column:
                data[column] = data[column].apply(lambda x: literal_eval(x.replace('nan', 'None')))

        head_entity_columns = [column for column in data.columns if "head_entity" in column]
        relation_columns = [column.replace("head_entity", "relation") for column in head_entity_columns]
        tail_entity_columns = [column for column in data.columns if column.startswith("tail_entities")]
        data.head()


        for index in tqdm(range(data.shape[0]), disable=True):
            df_row_pruned = prune(data.iloc[[index]], filename=filename, max_tail_entities=1)
            start_time = time()
            context = get_context(df_row_pruned.iloc[0]['subgraph'], 
                                target_head_entities=list(df_row_pruned .iloc[0][head_entity_columns].values),
                                target_relations=list(df_row_pruned .iloc[0][relation_columns].values),
                                target_tail_entities=list(set(chain.from_iterable(df_row_pruned.iloc[0][tail_entity_columns].values))),
                                relation_separator="/" if "FB15k" in filename else (":" if "NELL" in filename else (" " if "wiki" in filename else None)),
                                tokenizer=llm.get_tokenizer(),
                                max_context_len=args.context_length,
                                verbose=False)
            context.drop_duplicates(keep='first', inplace=True)
            time_context = time() - start_time

            if("FB15k" in filename):
                context_as_string = ("\n").join([(" | ").join([head_entity_context, relation_context.split("/")[-1], tail_entity_context])
                                            for head_entity_context, relation_context, tail_entity_context in context.values]).replace("_", " ")  
            elif("NELL" in filename):
                context_as_string = ("\n").join([(" | ").join([head_entity_context, relation_context.split(":")[-1], tail_entity_context])
                                            for head_entity_context, relation_context, tail_entity_context in context.values]).replace("_", " ") 
            else:
                context_as_string = ("\n").join([(" | ").join([head_entity_context, relation_context, tail_entity_context])
                                            for head_entity_context, relation_context, tail_entity_context in context.values]).replace("_", " ") 
            
            if(len(llm.get_tokenizer().encode(context_as_string)) > 2*args.context_length):
                logger.info("Warning: Context too long", index)
                continue

            

            df_row_pruned.drop(columns=['subgraph'], inplace=True)
            df_row_exploded = df_row_pruned.explode(['tail_entities', 'flipped_tail_entities'])
            df_row_exploded.rename(columns={'tail_entities': 'tail_entity', 'flipped_tail_entities': 'flipped_tail_entity'}, inplace=True)
            df_row_exploded.dropna(inplace=True)
            # df_row_exploded


            instruction = "Consider the context as a set of triplets where entries are separated by '|' symbol. Answer question according to the context.\n\n"
            question_instruction_single_chain = "Do not add additional text. Is the following triplet FACTUALLY CORRECT? Answer with Yes or No.\n\n"
            
            for i in range(df_row_exploded.shape[0]):
                    
                    # check the context contains the tail entity, head entity and relation in an specific row
                    mask = (context['tail_entity'] == df_row_exploded.iloc[i]['tail_entity']) & \
                            (context['head_entity'] == df_row_exploded.iloc[i]['head_entity']) & \
                            (context['relation'] == df_row_exploded.iloc[i]['relation'])
                    if np.sum(mask) != 1:
                        logger.info(f"Warning: Assertion false for index {index + (chunkidx - 1) * chunksize}")
                        continue
                    
                    
                    # flipped tail entitiy
                    mask = (context['tail_entity'] == df_row_exploded.iloc[i]['flipped_tail_entity']) & \
                            (context['head_entity'] == df_row_exploded.iloc[i]['head_entity']) & \
                            (context['relation'] == df_row_exploded.iloc[i]['relation'])
                    
                    if np.sum(mask) != 0:
                        logger.info(f"Warning: Assertion false for index {index + (chunkidx - 1) * chunksize}")
                        continue

                
                        
                    if("FB15k" in filename):
                        target_triplet = (" | ").join([df_row_exploded.iloc[i]['head_entity'], 
                                                    df_row_exploded.iloc[i]['relation'].split("/")[-1], 
                                                    df_row_exploded.iloc[i]['tail_entity']]).replace("_", " ")
                        flipped_target_triplet = (" | ").join([df_row_exploded.iloc[i]['head_entity'], 
                                                    df_row_exploded.iloc[i]['relation'].split("/")[-1], 
                                                    df_row_exploded.iloc[i]['flipped_tail_entity']]).replace("_", " ")

                    elif("NELL" in filename):
                        target_triplet = (" | ").join([df_row_exploded.iloc[i]['head_entity'], 
                                                    df_row_exploded.iloc[i]['relation'].split(":")[-1], 
                                                    df_row_exploded.iloc[i]['tail_entity']]).replace("_", " ")
                        flipped_target_triplet = (" | ").join([df_row_exploded.iloc[i]['head_entity'], 
                                                    df_row_exploded.iloc[i]['relation'].split(":")[-1], 
                                                    df_row_exploded.iloc[i]['flipped_tail_entity']]).replace("_", " ")
                    else:
                        target_triplet = (" | ").join([df_row_exploded.iloc[i]['head_entity'], 
                                                    df_row_exploded.iloc[i]['relation'], 
                                                    df_row_exploded.iloc[i]['tail_entity']]).replace("_", " ")
                        flipped_target_triplet = (" | ").join([df_row_exploded.iloc[i]['head_entity'], 
                                                    df_row_exploded.iloc[i]['relation'], 
                                                    df_row_exploded.iloc[i]['flipped_tail_entity']]).replace("_", " ")

                    for use_context in [False, True]:
                        result = {
                            "use_context": use_context,
                            "prompt_base_query" : None,
                            "raw_response_base_query" : None,
                            "response_base_query" : None,
                            "ground_truth_base_query" : None,
                            "correct_base_query" : None,
                            "logically_consistent": None,
                            "prompt_negation_query" : None,
                            "raw_response_negation_query" : None,
                            "response_negation_query" : None,
                            "ground_truth_negation_query" : None,
                            "correct_negation_query" : None,
                            "time_context": None,
                            "time_response_base_query": None,
                            "time_response_negation_query": None,
                        }
                        # add previous time info
                        for column_df_row_pruned in df_row_pruned.columns:
                            if(column_df_row_pruned.startswith("time")):
                                result[column_df_row_pruned] = df_row_pruned.iloc[0][column_df_row_pruned]

                        # base query
                        if(use_context):
                            result['time_context'] = time_context
                            prompt = instruction + \
                                    context_as_string + "\n\n" + \
                                    question_instruction_single_chain + \
                                    target_triplet
                        else:
                            prompt = question_instruction_single_chain + \
                                        target_triplet

                        result['prompt_base_query'] = prompt
                        result["ground_truth_base_query"] = True
                        
                            
                        
                        # negation query
                        if(use_context):
                            prompt = instruction + \
                                    context_as_string + "\n\n" + \
                                    question_instruction_single_chain + \
                                    flipped_target_triplet
                        else:
                            prompt = question_instruction_single_chain + \
                                        flipped_target_triplet

                        result['prompt_negation_query'] = prompt
                        result["ground_truth_negation_query"] = False
                         
                        

                        
                        # replace with newline
                        for key in result.keys():
                            if(result[key] is not None and ("prompt" in key or "raw_response" in key)):
                                # print(key)
                                result[key] = result[key].replace('\n', '[NEWLINE]')
                        
                        # store results via append
                        result_df = pd.DataFrame([result])
                        if(not os.path.exists(f'{save_path}/{store_filename}')):
                            result_df.to_csv(f'{save_path}/{store_filename}', index=False)
                        else:
                            result_df.to_csv(f'{save_path}/{store_filename}', index=False, header=False, mode='a')