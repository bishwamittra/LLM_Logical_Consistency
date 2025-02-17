
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import numpy as np
from time import time
import os
import argparse
from vllm import LLM, SamplingParams
import json
import sys
sys.path.append("../prompt_generation/")
from utils_llm_consistency import get_context, prune
from utils import get_logger


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--nrows", type=int, default=None)
parser.add_argument("--model_name", type=str, default="llama7b_chat_hf")
parser.add_argument("--cuda_id", type=int, default=1)
parser.add_argument("--chunksize", type=int, default=10, help="Chunksize for processing the data")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--save_root", type=str, default="result", help="Root directory to save the results")
parser.add_argument("--without_context", action="store_true", default=False, help="Compute responses where context is not provided")



# finetuned model
parser.add_argument("--use_checkpoint", action="store_true", default=False, help="Use checkpoint for the finetuned model")
parser.add_argument("--finetuned_model_path", type=str, default=None, help="Path to the finetuned model")
parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint number for the finetuned model")

args = parser.parse_args()


save_root = args.save_root
if(args.use_checkpoint):
    assert args.checkpoint is not None
    logger, exp_seq, save_path = get_logger(save_root=save_root, save_tag=f"complex_query_consistency_on_checkpoint_{args.checkpoint}")
else:
    logger, exp_seq, save_path = get_logger(save_root=save_root, save_tag="complex_query_consistency")
logger.info(f"=======Exp: {exp_seq}=============")
logger.info(f"Model: {args.model_name}")



config = None
with open("../_config.json") as f:
    config = json.load(f)
if(config == None):
    quit()

if(args.use_checkpoint):
        assert args.finetuned_model_path is not None
        llm = LLM(model=args.finetuned_model_path, tensor_parallel_size=args.num_gpus)

else:
    llm = LLM(model=config[args.model_name]['model_path'], tensor_parallel_size=args.num_gpus)
sampling_params = SamplingParams(temperature=config[args.model_name]['temperature'], 
                                 max_tokens=config[args.model_name]['max_tokens'],
                                 top_k=config[args.model_name]['top_k'])



filename = args.filename
assert "prompt" in filename
if("/" in filename):
    store_filename = f"{filename.split('/')[-1]}".replace("prompt", f"{args.model_name}_response")
else:
    store_filename = f"{filename}".replace("prompt", f"{args.model_name}_response")
logger.info(f"Store filename: {store_filename}")
assert "/" not in store_filename
if(args.nrows is not None):
    total_lines = args.nrows
else:
    total_lines = sum(1 for row in open(filename, 'r'))
logger.info(f"Total lines in {filename}: {total_lines}")
chunksize = args.chunksize
chunkidx = 0
total_chunks = total_lines//chunksize + 1
logger.info(f"Chunksize: {chunksize}")
logger.info(f"Total chunks: {total_chunks}")
with pd.read_csv(filename, chunksize=chunksize, nrows=args.nrows) as reader:
    for data in reader:
        if(not args.without_context):
            data = data[data['use_context'] == True]
        data = data.reset_index()
        logger.info(f"Loaded chunk {chunkidx + 1}/{total_chunks} from {filename} with shape {data.shape}")
        chunkidx += 1

        for column in data.columns:
            if("prompt" in column):
                data[column] = data[column].apply(str)
                data[column] = data[column].apply(lambda x: x.replace('[NEWLINE]', '\n'))
        
        num_subqueries = 0
        for column in data.columns:
            if("prompt_subquery" in column):
                num_subqueries += 1
        


        # will store this dataframe
        data_updated = data.copy()
        data_updated.reset_index(inplace=True)


        if(args.model_name in ["llama7b_chat_hf", "llama13b_chat_hf"]):
            data['prompt_base_query'] = data['prompt_base_query'].apply(lambda x:
                f"<s><<SYS>>You are a helpful assistant. Always follow the intstructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {x} [/INST]"
            )
            for j in range(num_subqueries):
                data[f"prompt_subquery_{j+1}"] = data[f"prompt_subquery_{j+1}"].apply(lambda x:
                    f"<s><<SYS>>You are a helpful assistant. Always follow the intstructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {x} [/INST]"
                )
        elif(args.model_name == "gemma-2b-it"):
            data['prompt_base_query'] = data['prompt_base_query'].apply(lambda x:
                f"<bos><start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model"
            )
            for j in range(num_subqueries):
                data[f"prompt_subquery_{j+1}"] = data[f"prompt_subquery_{j+1}"].apply(lambda x:
                    f"<bos><start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model"
                )
        
        else:
            raise NotImplementedError 

            
        start_time = time()
        response_list_base_query = []
        outputs = llm.generate(data['prompt_base_query'].values, sampling_params)
        for output in outputs:
            response_list_base_query.append(output.outputs[0].text)

        
        data_updated['time_response_base_query'] = (time() - start_time) / data_updated.shape[0]
        data_updated['raw_response_base_query'] = response_list_base_query
        data_updated['response_base_query'] = data_updated['raw_response_base_query'].apply(lambda x: True if "yes" in x.lower() else False)
        data_updated["correct_base_query"] = data_updated.apply(lambda x: x['response_base_query'] if x['ground_truth_base_query'] else not x['response_base_query'], axis=1)


        # subqueries
        subquery_response = np.zeros(shape=(data_updated.shape[0], num_subqueries), dtype=bool)
        for j in range(num_subqueries):
            start_time = time()
            response_list_subquery = []
            response = llm.generate(data[f"prompt_subquery_{j+1}"].values, sampling_params)
            for output in response:
                response_list_subquery.append(output.outputs[0].text)
            data_updated[f"time_response_subquery_{j+1}"] = (time() - start_time) / data_updated.shape[0]

            data_updated[f"raw_response_subquery_{j+1}"] = response_list_subquery
            data_updated[f"response_subquery_{j+1}"] = data_updated[f"raw_response_subquery_{j+1}"].apply(lambda x: True if "yes" in x.lower() else False)
            data_updated[f"correct_subquery_{j+1}"] = data_updated.apply(lambda x: x[f"response_subquery_{j+1}"] if x[f"ground_truth_subquery_{j+1}"] else not x['response_base_query'], axis=1)

            subquery_response[:, j] = data_updated[f"response_subquery_{j+1}"].values  
 
        # print(subquery_response)

        if("1u2i_prompt" in args.filename):
            second_branch_verdict = np.sum(subquery_response[:, 1:], axis=1) == num_subqueries - 1
            subquery_verdict = np.logical_or(subquery_response[:, 0], second_branch_verdict)
        elif("1i2u_prompt" in args.filename):
            second_branch_verdict = np.sum(subquery_response[:, 1:], axis=1) > 0
            subquery_verdict = np.logical_and(subquery_response[:, 0], second_branch_verdict)
        elif("u_prompt" in args.filename):
            subquery_verdict = np.sum(subquery_response, axis=1) > 0
        elif("i_prompt" in args.filename):
            subquery_verdict = np.sum(subquery_response, axis=1) == num_subqueries
        else:
            raise ValueError("Query type not supported")

        # print(subquery_verdict)


        for i in range(data_updated.shape[0]):
            data_updated.loc[i, 'logically_consistent'] = True if data_updated.loc[i, 'response_base_query'] == subquery_verdict[i] else False
        
        # replace with newline
        for column in data_updated.columns:
            if("prompt" in column or "raw_response" in column):
                data_updated[column] = data_updated[column].apply(str)
                data_updated[column] = data_updated[column].apply(lambda x: x.replace('\n', '[NEWLINE]'))

        
        # store results via append
        if(not os.path.exists(f'{save_path}/{store_filename}')):
            data_updated.to_csv(f'{save_path}/{store_filename}', index=False)
        else:
            data_updated.to_csv(f'{save_path}/{store_filename}', index=False, header=False, mode='a')
        
        
        