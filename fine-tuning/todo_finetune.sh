model="base_models/Llama-2-7b-chat-hf"
# model="base_models/Llama-2-13b-chat-hf"
# model="base_models/gemma-2b-it"

# ls ${model}

output_dir="results_$(date +%Y_%m_%d_%H_%M)_learning_rate_2e-6_without_subqueries_2000_samples_per_query_gemma_2b_it_shuffled"
python finetune.py \
--model_path ${model} \
--train_filename ../data_optimized/prompts/1c_prompt_FB15k_train_2.csv ../data_optimized/prompts/2i_prompt_FB15k_train_2.csv ../data_optimized/prompts/2u_prompt_FB15k_train_2.csv \
--eval_filename ../data_optimized/prompts/1c_prompt_FB15k_valid_2.csv ../data_optimized/prompts/2i_prompt_FB15k_valid_2.csv ../data_optimized/prompts/2u_prompt_FB15k_valid_2.csv \
--query_type 1c 2i 2u \
--num_train_epochs 20 \
--train_nrows 2000 \
--eval_nrows 5000 \
--gradient_checkpointing \
--group_by_length \
--max_seq_length 4096 \
--output_dir ${output_dir} \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--learning_rate 2e-6 \
--use_4bit \
