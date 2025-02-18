
import pandas as pd
import os
import json
from vllm import LLM, SamplingParams
from time import time
import argparse
import sys
sys.path.append("../prompt_generation/")
from utils import get_logger

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--nrows", type=int, default=None)
parser.add_argument("--model_name", type=str, default="llama7b_chat_hf")
parser.add_argument("--chunksize", type=int, default=10, help="Chunksize for processing the data")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--save_root", type=str, default="result", help="Root directory to save the results")
parser.add_argument("--without_context", action="store_true", default=False, help="Compute responses where context is not provided")

# finetuned model
parser.add_argument("--use_checkpoint", action="store_true", default=False, help="Use checkpoint for the finetuned model")
parser.add_argument("--finetuned_model_path", type=str, default=None, help="Path to the finetuned model")
parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint number for the finetuned model")
# parser.add_argument("--cuda_id", type=int, default=1)
args = parser.parse_args()




save_root = args.save_root
if(args.use_checkpoint):
    assert args.checkpoint is not None
    logger, exp_seq, save_path = get_logger(save_root=save_root, save_tag=f"syl-fol_consistency_on_checkpoint_{args.checkpoint}")
else:
    logger, exp_seq, save_path = get_logger(save_root=save_root, save_tag="syl-fol_consistency")
logger.info(f"=======Exp: {exp_seq}=============")
logger.info(f"Model: {args.model_name}")






config = None
with open("../config.json") as f:
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
logger.info(f"Total lines considered in {filename}: {total_lines}")
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


        # will store this dataframe
        data_updated = data.copy()
        data_updated.reset_index(inplace=True)
            
            
        if(args.model_name in ["llama7b_chat_hf", "llama13b_chat_hf", "llama70b_chat_hf"]):
            data['prompt_base_query'] = data['prompt_base_query'].apply(lambda x:
                f"<s><<SYS>>You are a helpful assistant. Always follow the intstructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {x} [/INST]"
            )
            if "syllogism" in filename:
                data['prompt_syl_query'] = data['prompt_syl_query'].apply(lambda x:
                    f"<s><<SYS>>You are a helpful assistant. Always follow the intstructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {x} [/INST]"
                )
            elif "exists-forall" in filename:
                data['prompt_negation_query'] = data['prompt_negation_query'].apply(lambda x:
                    f"<s><<SYS>>You are a helpful assistant. Always follow the intstructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {x} [/INST]"
                )
            elif "de-morgan" in filename:
                data['prompt_de-morgan_query'] = data['prompt_de-morgan_query'].apply(lambda x:
                    f"<s><<SYS>>You are a helpful assistant. Always follow the intstructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {x} [/INST]"
                )
        elif(args.model_name == "gemma-2b-it"):
            data['prompt_base_query'] = data['prompt_base_query'].apply(lambda x:
                f"<bos><start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model"
            )
            if "syllogism" in filename:
                data['prompt_syl_query'] = data['prompt_syl_query'].apply(lambda x:
                    f"<bos><start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model"
                )
            elif "exists-forall" in filename:
                data['prompt_negation_query'] = data['prompt_negation_query'].apply(lambda x:
                    f"<bos><start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model"
                )
            elif "de-morgan" in filename:
                data['prompt_de-morgan_query'] = data['prompt_de-morgan_query'].apply(lambda x:
                    f"<bos><start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model"
                )
        elif args.model_name == "vicuna68m":
            pass
        else:
            raise NotImplementedError
            
                
                
        # base query
        start_time = time()        
        response_list_base_query = []
        outputs = llm.generate(data['prompt_base_query'].values, sampling_params)
        for output in outputs:
            response_list_base_query.append(output.outputs[0].text)
    
        data_updated['time_response_base_query'] = (time() - start_time) / data_updated.shape[0]
        data_updated['raw_response_base_query'] = response_list_base_query
        data_updated['response_base_query'] = data_updated['raw_response_base_query'].apply(lambda x: True if "yes" in x.lower() else False)
        data_updated["correct_base_query"] = data_updated.apply(lambda x: x['response_base_query'] if x['ground_truth_base_query'] else not x['response_base_query'], axis=1)


        # syl query
        start_time = time()
        response_list_syl_fol_query = []
        if "syllogism" in filename:
            outputs = llm.generate(data['prompt_syl_query'].values, sampling_params)
        elif "exists-forall" in filename:
            outputs = llm.generate(data['prompt_negation_query'].values, sampling_params)
        elif "de-morgan" in filename:
            outputs = llm.generate(data['prompt_de-morgan_query'].values, sampling_params)
        for output in outputs:
            response_list_syl_fol_query.append(output.outputs[0].text)

        if "syllogism" in filename:
            data_updated['time_response_syl_query'] = (time() - start_time) / data_updated.shape[0]
            data_updated['raw_response_syl_query'] = response_list_syl_fol_query
            data_updated['response_syl_query'] = data_updated['raw_response_syl_query'].apply(lambda x: False if "yes" not in x.lower() else True)
            data_updated["correct_syl_query"] = data_updated.apply(lambda x: x['response_syl_query'] if x['ground_truth_syl_query'] else not x['response_syl_query'], axis=1)
            data_updated['logically_consistent'] = data_updated.apply(lambda x: True if x['response_base_query'] == x['response_syl_query'] else False, axis=1)
        elif "exists-forall" in filename:
            data_updated['time_response_negation_query'] = (time() - start_time) / data_updated.shape[0]
            data_updated['raw_response_negation_query'] = response_list_syl_fol_query
            data_updated['response_negation_query'] = data_updated['raw_response_negation_query'].apply(lambda x: False if "yes" not in x.lower() else True)
            data_updated['correct_negation_query'] = data_updated.apply(lambda x: x['response_negation_query'] if x['ground_truth_negation_query'] else not x['response_negation_query'], axis=1)
            data_updated['logically_consistent'] = data_updated.apply(lambda x: True if x['response_base_query'] != x['response_negation_query'] else False, axis=1)
        elif "de-morgan" in filename:
            data_updated['time_response_de-morgan_query'] = (time() - start_time) / data_updated.shape[0]
            data_updated['raw_response_de-morgan_query'] = response_list_syl_fol_query
            data_updated['response_de-morgan_query'] = data_updated['raw_response_de-morgan_query'].apply(lambda x: False if "yes" not in x.lower() else True)
            data_updated['correct_de-morgan_query'] = data_updated.apply(lambda x: x['response_de-morgan_query'] if x['ground_truth_de-morgan_query'] else not x['response_de-morgan_query'], axis=1)
            data_updated['logically_consistent'] = data_updated.apply(lambda x: True if x['response_base_query'] == x['response_de-morgan_query'] else False, axis=1)
        else:
            raise ValueError(filename)



                
        
        
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
        
        
# if(args.use_checkpoint):
#     # delete the temporary model
#     os.system(f"rm -r {args.finetuned_model_path}")