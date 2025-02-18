
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import numpy as np
from time import time
import os
from utils_llm_consistency import get_context, prune
import argparse
from utils import get_logger
from vllm import LLM, SamplingParams
import json
from itertools import chain, product


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--model_name", type=str, default="llama7b_chat_hf", choices=["llama13b_hf", "llama13b_chat_hf", "llama7b_hf", "llama7b_chat_hf"])
parser.add_argument("--context_length", type=int, default=1000)
parser.add_argument("--nrows", type=int, default=None)
parser.add_argument("--cuda_id", type=int, default=1)
args = parser.parse_args()


save_root = "result"
logger, exp_seq, save_path = get_logger(save_root=save_root, save_tag="complex_query_consistency")
logger.info(f"=======Exp: {exp_seq}=============")
logger.info(f"Model: {args.model_name}")

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)

model_name = args.model_name


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
assert "/" in filename
store_filename = f"{filename.split('/')[-1].replace('data_final', 'prompt')}"
assert "/" not in store_filename
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

        
        num_subqueries = 0
        for column in data.columns:
            if("flipped_tail_entities" in column and column != "flipped_tail_entities"):
                num_subqueries += 1
        num_subqueries

        head_entity_columns = [column for column in data.columns if "head_entity" in column]
        relation_columns = [column.replace("head_entity", "relation") for column in head_entity_columns]
        tail_entity_columns = [column for column in data.columns if column.startswith("tail_entities")]
        
        for index in tqdm(range(data.shape[0]), disable=True):
            # prune
            df_row_pruned = prune(data.iloc[[index]], filename=filename, max_tail_entities=1)

            
            # get context
            start_time = time()
            context = get_context(df_row_pruned.iloc[0]['subgraph'], 
                                target_head_entities=list(df_row_pruned.iloc[0][head_entity_columns].values),
                                target_relations=list(df_row_pruned.iloc[0][relation_columns].values),
                                target_tail_entities=list(set(chain.from_iterable(df_row_pruned.iloc[0][tail_entity_columns].values))),
                                relation_separator="/" if "FB15k" in filename else (":" if "NELL" in filename else None),
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
                raise ValueError(f"{filename} not recognzied") 
            
            if(len(llm.get_tokenizer().encode(context_as_string)) > 2*args.context_length):
                print("Warning: Context too long", index)
                print(context)
                continue


            

            instruction = "Consider the context as a set of triplets where entries are separated by '|' symbol. Answer question according to the context.\n\n"
            question_instruction_single_chain = "Do not add additional text. Is the following logic query FACTUALLY CORRECT? Answer with Yes or No.\n\n"

            inconsistent_data = False
            # empty tail entities
            for i in range(num_subqueries):
                # print(i, df_row_pruned.iloc[0][f'tail_entities_{i+1}'], df_row_pruned.iloc[0][f'flipped_tail_entities_{i+1}'])
                if(len(df_row_pruned.iloc[0][f'tail_entities_{i+1}']) == 0):
                    inconsistent_data = True
                    break
                if(None in df_row_pruned.iloc[0][f'tail_entities_{i+1}']):
                    inconsistent_data = True
                    break
                if(None in df_row_pruned.iloc[0][f'flipped_tail_entities_{i+1}']):
                    inconsistent_data = True
                    break
                assert len(df_row_pruned.iloc[0][f'tail_entities_{i+1}']) == 1
                assert len(df_row_pruned.iloc[0][f'flipped_tail_entities_{i+1}']) == 1

            if(inconsistent_data):
                continue


            unique_tail_entities = []
            for i in range(num_subqueries):
                unique_tail_entities.append(df_row_pruned[f'tail_entities_{i+1}'].item() + df_row_pruned[f'flipped_tail_entities_{i+1}'].item())
            # print(unique_tail_entities)
            relations = []
            for i in range(num_subqueries):
                relations.append(df_row_pruned[f'relation_{i+1}'].item())
            


            for use_context in [True, False]:
                
                for tail_entity_tuple in list(product(*unique_tail_entities)):
                    result = {
                        "use_context" : use_context,
                        'prompt_base_query' : None,
                        "raw_response_base_query" : None,
                        "response_base_query" : None,
                        "ground_truth_base_query" : None,
                        "correct_base_query" : None,
                        "logically_consistent": None,
                        "time_context": None,
                        "time_response_base_query": None,
                    }

                    for i in range(num_subqueries):
                        result[f"prompt_subquery_{i+1}"] = None
                        result[f"raw_response_subquery_{i+1}"] = None
                        result[f"response_subquery_{i+1}"] = None
                        result[f"ground_truth_subquery_{i+1}"] = None
                        result[f"correct_subquery_{i+1}"] = None
                        result[f"time_response_subquery_{i+1}"] = None

                    # add previous time info
                    for column_df_row_pruned in df_row_pruned.columns:
                        if(column_df_row_pruned.startswith("time")):
                            result[column_df_row_pruned] = df_row_pruned.iloc[0][column_df_row_pruned]

                        
                    test_triplets = []
                    test_triplets_raw = []
                    if("FB15k" in filename):
                        for i in range(num_subqueries):
                            test_triplets.append((" | ").join((df_row_pruned[f'head_entity_{i+1}'].item(), relations[i].split("/")[-1], tail_entity_tuple[i])).replace("_", " "))
                            test_triplets_raw.append((df_row_pruned[f'head_entity_{i+1}'].item(), relations[i], tail_entity_tuple[i]))
                    elif("NELL" in filename):
                        for i in range(num_subqueries):
                            test_triplets.append((" | ").join((df_row_pruned[f'head_entity_{i+1}'].item(), relations[i].split(":")[-1], tail_entity_tuple[i])).replace("_", " "))
                            test_triplets_raw.append((df_row_pruned[f'head_entity_{i+1}'].item(), relations[i], tail_entity_tuple[i]))
                    else:
                        raise ValueError(f"{filename} not recognzied")

                    # base prompt
                    if("1u2i_data_final" in filename):
                        prompt =  question_instruction_single_chain + \
                            f"( {test_triplets[0]} )" + "  OR " + \
                            "\n(\n" + \
                            (" AND \n").join([f"( {test_triplet} )" for test_triplet in test_triplets[1:]]) + \
                            "\n)"
                    elif("1i2u_data_final" in filename):
                        prompt =  question_instruction_single_chain + \
                            f"( {test_triplets[0]} )" + "  AND " + \
                            "\n(\n" + \
                            (" OR \n").join([f"( {test_triplet} )" for test_triplet in test_triplets[1:]]) + \
                            "\n)"
                    elif("i_data_final" in filename):
                        prompt =  question_instruction_single_chain + \
                            (" AND \n").join([f"( {test_triplet} )" for test_triplet in test_triplets])
                    elif("u_data_final" in filename):
                        prompt =  question_instruction_single_chain + \
                            (" OR \n").join([f"( {test_triplet} )" for test_triplet in test_triplets])
                    if(use_context):
                        result['time_context'] = time_context
                        prompt = instruction + \
                            context_as_string + "\n\n" + prompt
                    
                    result["prompt_base_query"] = prompt
                    


                        

                    inconsistent_context = False
                    for i in range(num_subqueries):

                        if(use_context):
                            prompt = instruction + \
                                context_as_string + "\n\n" + \
                                question_instruction_single_chain + \
                                test_triplets[i]

                        else:
                            prompt = question_instruction_single_chain + \
                                test_triplets[i]

                        result[f"prompt_subquery_{i+1}"] = prompt

                        
                        mask = (context['head_entity'] == test_triplets_raw[i][0]) & \
                                (context['relation'] == test_triplets_raw[i][1]) & \
                                (context['tail_entity'] == test_triplets_raw[i][2])
                        
                        if(tail_entity_tuple[i] in df_row_pruned[f'tail_entities_{i+1}'].item()):
                            result[f"ground_truth_subquery_{i+1}"] = True
                            if(np.sum(mask) != 1):
                                inconsistent_context = True
                                break
                        else:
                            result[f"ground_truth_subquery_{i+1}"] = False
                            if(np.sum(mask) != 0):
                                inconsistent_context = True
                                break
                    if(inconsistent_context):
                        logger.info(f"Warning: Assertion false for index {index + (chunkidx - 1) * chunksize}")
                        continue

                    # base query ground truth
                    if("1u2i_data_final" in filename or "1i2u_data_final" in filename):
                        assert num_subqueries == 3
                        first_relation_verdict = result[f"ground_truth_subquery_{1}"]
                        second_relation_verdict = result[f"ground_truth_subquery_{2}"]
                        third_relation_verdict = result[f"ground_truth_subquery_{3}"]

                        if("1u2i_data_final" in filename):
                            # disjunction followed by two conjunction rules
                            if(first_relation_verdict or (second_relation_verdict and third_relation_verdict)):
                                result["ground_truth_base_query"] = True
                            else:
                                result["ground_truth_base_query"] = False

                        elif("1i2u_data_final" in filename):
                            # conjunction followed by two disjunction rules
                            if(first_relation_verdict and (second_relation_verdict or third_relation_verdict)):
                                result["ground_truth_base_query"] = True
                            else:
                                result["ground_truth_base_query"] = False
                    else:
                        sum_subquery = 0
                        for i in range(num_subqueries):
                            sum_subquery += result[f"ground_truth_subquery_{i+1}"]
                        if("u_data_final" in filename):
                            if(sum_subquery > 0):
                                result["ground_truth_base_query"] = True
                                
                            else:
                                result["ground_truth_base_query"] = False
                        elif("i_data_final" in filename):
                            if(sum_subquery == num_subqueries):
                                result["ground_truth_base_query"] = True
                            else:
                                result["ground_truth_base_query"] = False
                    
                                      
                    
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
            