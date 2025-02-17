import argparse
import pandas as pd
import numpy as np
from datasets import Dataset
# from transformers import TrainerCallback

import bitsandbytes as bnb

def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str,default="NousResearch/Llama-2-7b-chat-hf")
    parser.add_argument("--model_path", type=str, help="model path", default="base_models/Llama-2-7b-chat-hf")

    parser.add_argument("--train_filename", 
                        type=str, 
                        default=[],
                        nargs='+',
                        help="training data path [a list]")
    parser.add_argument("--eval_filename",
                        type=str,
                        default=[],
                        nargs='+',
                        help="evaluation data path [a list]")
    parser.add_argument("--query_type", 
                        type=str, 
                        default=[],
                        nargs='+',
                        help="query type [a list]")   
    parser.add_argument("--train_nrows", type=int, default=5000, help="number of rows to read from training data")
    parser.add_argument("--eval_nrows", type=int, default=None, help="number of rows to read from evaluation data")


    # Fine-tuned model name
    parser.add_argument("--model_path_fine_tuned",type=str,default = "finetuned_models/fine_tuned_model")

    ################################################################################
    # QLoRA parameters
    ################################################################################
    # LoRA attention dimension
    parser.add_argument("--lora_r",type=int,default = 64)

    # Alpha parameter for LoRA scaling
    parser.add_argument("--lora_alpha",type=int,default=16)

    # Dropout probability for LoRA layers
    parser.add_argument("--lora_dropout",type=float,default= 0.1)

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    parser.add_argument("--use_4bit", action='store_true', default=False)

    # Compute dtype for 4-bit base models
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default= "float16")

    # Quantization type (fp4 or nf4)
    parser.add_argument("--bnb_4bit_quant_type",type=str,default= "nf4")

    # Activate nested quantization for 4-bit base models (double quantization)
    parser.add_argument("--use_nested_quant",action="store_true", default= False)

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    parser.add_argument("--output_dir",type=str, default= "./results")

    # Number of training epochs
    parser.add_argument("--num_train_epochs",type=int,default = 2)

    # Batch size per GPU for training
    parser.add_argument("--per_device_train_batch_size",type=int,default = 4)

    # Batch size per GPU for evaluation
    parser.add_argument("--per_device_eval_batch_size",type=int,default = 4)

    # Number of update steps to accumulate the gradients for
    parser.add_argument("--gradient_accumulation_steps",type=int,default = 1)

    # Enable gradient checkpointing
    parser.add_argument("--gradient_checkpointing",action='store_true',default = False)

    # Maximum gradient normal (gradient clipping)
    parser.add_argument("--max_grad_norm", type=float, default = 0.3)

    # Initial learning rate (AdamW optimizer)
    parser.add_argument("--learning_rate",type=float,default = 2e-4)

    # Weight decay to apply to all layers except bias/LayerNorm weights
    parser.add_argument("--weight_decay",type=float,default = 0.001)

    # Optimizer to use
    parser.add_argument("--optim",type=str, default= "paged_adamw_32bit")

    # Learning rate schedule
    parser.add_argument("--lr_scheduler_type",type=str,default = "cosine")

    # Number of training steps (overrides num_train_epochs)
    parser.add_argument("--max_steps",type=int,default = -1)

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    parser.add_argument("--warmup_ratio",type=float,default = 0.03)

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    parser.add_argument("--group_by_length",action="store_true",default = False)

    # Save checkpoint every X updates steps
    parser.add_argument("--save_steps",type=int,default=0)

    # Log every X updates steps
    parser.add_argument("--logging_steps",type=int,default= 25)

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    parser.add_argument("--max_seq_length",type=int,default = None)

    # Pack multiple short examples in the same input sequence to increase efficiency
    parser.add_argument("--packing",action="store_true", default = False)


    return parser.parse_args()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataset(filenames: str,
                query_types: str,
                model_path: str,
                nrows: int = None
    ) -> Dataset:

    if(len(filenames) == 0):
        raise ValueError("No filename provided")
    assert len(filenames) == len(query_types), "Number of filenames and query types should be the same"
    for i, filename in enumerate(filenames):
        assert query_types[i] in filename or query_types[i] in ['exists_forall', 'syllogism'], f"Query type {query_types[i]} not supported for filename {filename}"

    result = {
        "prompt": []
    }
    # sys_instruction = "<<SYS>>\nClassify the query as Yes or No.\n<</SYS>>\n\n"
    sys_instruction = ""
    for filename, query_type in zip(filenames, query_types):
        df = pd.read_csv(filename)
        df = df[df['use_context'] == True] # only get prompts with context
        if df.shape[0] > nrows:
            df = df.sample(n=nrows, random_state=42)
        newline = "\n"
        if(query_type == "1c"):
            base_query_column = 'prompt_base_query'
            negation_query_column = 'prompt_negation_query'

            if("Llama" in model_path):
                df[base_query_column] = df[base_query_column].apply(lambda x: f"<s>[INST]{sys_instruction}{x.replace('[NEWLINE]', newline)}[/INST] Yes</s>")
                df[negation_query_column] = df[negation_query_column].apply(lambda x: f"<s>[INST]{sys_instruction}{x.replace('[NEWLINE]', newline)}[/INST] No</s>")
            
            elif("gemma" in model_path):
                df[base_query_column] = df.apply(lambda x: f"<bos><start_of_turn>user\n{x[base_query_column].replace('[NEWLINE]', newline)}<end_of_turn>\n<start_of_turn>model\nYes, because {x[base_query_column].replace('[NEWLINE]', newline).split(newline)[-1]} is present in the context.<end_of_turn>", axis=1)
                df[negation_query_column] = df.apply(lambda x: f"<bos><start_of_turn>user\n{x[negation_query_column].replace('[NEWLINE]', newline)}<end_of_turn>\n<start_of_turn>model\nNo, because {x[negation_query_column].replace('[NEWLINE]', newline).split(newline)[-1]} is absent in the context.<end_of_turn>", axis=1)

            print(f"Inlcuding simple positive and negative query: {df.shape[0]*2}")
            for i in range(len(df)):
                result["prompt"].append(df[base_query_column].iloc[i])
                result["prompt"].append(df[negation_query_column].iloc[i])

        elif(query_type in ["exists_forall", "syllogism", "i-de-morgan", "u-de-morgan"]):
            base_query_column = 'prompt_base_query'
            if query_type == "exists_forall":
                other_query_column = 'prompt_negation_query'
            elif query_type == "syllogism":
                other_query_column = 'prompt_syl_query'
            else:
                other_query_column = "prompt_de-morgan_query"

            ground_truth_base_column = base_query_column.replace("prompt", "ground_truth")
            ground_truth_other_column = other_query_column.replace("prompt", "ground_truth")

            if("Llama" in model_path):
                df[base_query_column] = df.apply(lambda x: f"<s>[INST]{sys_instruction}{x[base_query_column].replace('[NEWLINE]', newline)}[/INST] {'Yes' if x[ground_truth_base_column] else 'No'}</s>", axis=1)
                df[other_query_column] = df.apply(lambda x: f"<s>[INST]{sys_instruction}{x[other_query_column].replace('[NEWLINE]', newline)}[/INST] {'Yes' if x[ground_truth_other_column] else 'No'}</s>", axis=1)
            elif("gemma" in model_path):
                df[base_query_column] = df.apply(lambda x: f"<bos><start_of_turn>user\n{x[base_query_column].replace('[NEWLINE]', newline)}<end_of_turn>\n<start_of_turn>model\n{'Yes' if x[ground_truth_base_column] else 'No'}<end_of_turn>", axis=1)
                df[other_query_column] = df.apply(lambda x: f"<bos><start_of_turn>user\n{x[other_query_column].replace('[NEWLINE]', newline)}<end_of_turn>\n<start_of_turn>model\n{'Yes' if x[ground_truth_other_column] else 'No'}<end_of_turn>", axis=1)
            else:
                raise NotImplementedError()

            print(f"Inlcuding {query_type} queries: {df.shape[0]*2}")
            for i in range(len(df)):
                result["prompt"].append(df[base_query_column].iloc[i])
                result["prompt"].append(df[other_query_column].iloc[i])  



        elif(query_type in ["2i", "2u"]):
            num_subqueries = 2

            def get_instricution_answer(row, num_subqueries, query_type):
                verdict = "Yes" if row['ground_truth_base_query'] else "No"
                reasoning = []
                for i in range(num_subqueries):
                    reasoning.append(f"{row['prompt_subquery_'+str(i+1)].replace('[NEWLINE]', newline).split(newline)[-1]} is {'present' if row['ground_truth_subquery_'+str(i+1)] else 'absent'} in the context.")
                reasoning.append(f"Since the logical operator among subqueries is {'OR' if query_type == '2u' else 'AND'}, the answer is {verdict}.")

                return f"{verdict}. {newline}{newline}{newline.join(reasoning)}"

            base_query_column = 'prompt_base_query'
            # subquery_1_column = 'prompt_subquery_1'
            # subquery_2_column = 'prompt_subquery_2'

            if("Llama" in model_path):
                df[base_query_column] = df.apply(lambda x: f"<s>[INST]{sys_instruction}{x[base_query_column].replace('[NEWLINE]', newline)}[/INST] {'Yes' if x['ground_truth_base_query'] else 'No'}</s>", axis=1)
                # df[subquery_1_column] = df.apply(lambda x: f"<s>[INST]{sys_instruction}{x[subquery_1_column].replace('[NEWLINE]', newline)}[/INST] {'Yes' if x['ground_truth_subquery_1'] else 'No'}</s>", axis=1)
                # df[subquery_2_column] = df.apply(lambda x: f"<s>[INST]{sys_instruction}{x[subquery_2_column].replace('[NEWLINE]', newline)}[/INST] {'Yes' if x['ground_truth_subquery_2'] else 'No'}</s>", axis=1)

            elif("gemma" in model_path):
                # df[base_query_column] = df.apply(lambda x: f"<bos><start_of_turn>user\n{x[base_query_column].replace('[NEWLINE]', newline)}<end_of_turn>\n<start_of_turn>model\n{'Yes' if x['ground_truth_base_query'] else 'No'} <end_of_turn>", axis=1)
                df[base_query_column] = df.apply(lambda x: f"<bos><start_of_turn>user\n{x[base_query_column].replace('[NEWLINE]', newline)}<end_of_turn>\n<start_of_turn>model\n{get_instricution_answer(x, num_subqueries, query_type)} <end_of_turn>", axis=1)
                
                # df[subquery_1_column] = df.apply(lambda x: f"<bos><start_of_turn>user\n{x[subquery_1_column].replace('[NEWLINE]', newline)}<end_of_turn>\n<start_of_turn>model\n{'Yes' if x['ground_truth_subquery_1'] else 'No'} <end_of_turn>", axis=1)
                # df[subquery_2_column] = df.apply(lambda x: f"<bos><start_of_turn>user\n{x[subquery_2_column].replace('[NEWLINE]', newline)}<end_of_turn>\n<start_of_turn>model\n{'Yes' if x['ground_truth_subquery_2'] else 'No'} <end_of_turn>", axis=1)

            else:
                raise NotImplementedError()

            print(f"Inlcuding conjunctive and disjunctive queries: {df.shape[0]}")
            for i in range(len(df)):
                result["prompt"].append(df[base_query_column].iloc[i])
                # result["prompt"].append(df[subquery_1_column].iloc[i])
                # result["prompt"].append(df[subquery_2_column].iloc[i])
        else:
            raise ValueError(f"Query type {query_type} not supported")


    result_df = pd.DataFrame(result)
    # shuffle result_df
    # result_df = result_df.sample(frac=1).reset_index(drop=True)
    dataset_dict = {"text": result_df["prompt"].tolist()}
    dataset = Dataset.from_dict(dataset_dict)
    print(dataset)
    return dataset


# class ValidationCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, **kwargs):
#         print(f"Validation Loss: {state.log_metrics['eval_loss']}")

