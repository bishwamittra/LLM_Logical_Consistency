# toggle --use_4bit 

# all query types
output_dir="results_$(date +%Y_%m_%d_%H_%M)_learning_rate_2e-5_without_subqueries_2000_samples_per_query_llama7b"
python finetune.py \
--model_path base_models/Llama-2-7b-chat-hf \
--train_filename ../data_optimized/prompts/1c_prompt_FB15k_train_2.csv ../data_optimized/prompts/2i_prompt_FB15k_train_2.csv ../data_optimized/prompts/2u_prompt_FB15k_train_2.csv \
--eval_filename ../data_optimized/prompts/1c_prompt_FB15k_valid_2.csv ../data_optimized/prompts/2i_prompt_FB15k_valid_2.csv ../data_optimized/prompts/2u_prompt_FB15k_valid_2.csv \
--query_type 1c 2i 2u \
--num_train_epochs 2 \
--train_nrows 2000 \
--eval_nrows 5000 \
--gradient_checkpointing \
--group_by_length \
--max_seq_length 4096 \
--output_dir ${output_dir} \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--learning_rate 2e-5 \
--use_4bit \
