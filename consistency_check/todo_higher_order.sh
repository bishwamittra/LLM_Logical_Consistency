num_gpus=1
chunksize=500
nrows=5000


# for model_name in "gemma-2b-it" "llama7b_chat_hf" "llama13b_chat_hf"
# do 
#     save_root="result_$(date +%Y_%m_%d)_${model_name}_before_finetuning_temperature_0_rebuttal_de-morgan"

#     for filename in "../data_optimized/prompts/2i-de-morgan_prompt_FB15k_valid_2.csv" \
#                     "../data_optimized/prompts/2i-de-morgan_prompt_FB15k_test_2.csv" \
#                     "../data_optimized/prompts/2u-de-morgan_prompt_FB15k_valid_2.csv" \
#                     "../data_optimized/prompts/2u-de-morgan_prompt_FB15k_test_2.csv" \
#                     # "../data_optimized/prompts/2i-syllogism_prompt_FB15k_valid_2.csv" \
#                     # "../data_optimized/prompts/2i-syllogism_prompt_FB15k_test_2.csv" \
#                     # "../data_optimized/prompts/2i-exists-forall_prompt_FB15k_valid_2.csv" \
#                     # "../data_optimized/prompts/2i-exists-forall_prompt_FB15k_test_2.csv" \
                    
                    
#     do
#         wc -l $filename
#         time python consistency_check_syllogism_fol_batch_execution.py \
#         $filename \
#         --nrows  $nrows \
#         --chunksize $chunksize \
#         --num_gpus $num_gpus \
#         --model_name $model_name \
#         --save_root ${save_root}
#     done
# done


for model_name in "gemma-2b-it" "llama7b_chat_hf" "llama13b_chat_hf"
do 
    save_root="result_$(date +%Y_%m_%d)_${model_name}_before_finetuning_temperature_0_rebuttal_de-morgan_three_LLM_calls"

    for filename in "../data_optimized/prompts/2i-de-morgan_prompt_FB15k_valid_2.csv" \
                    "../data_optimized/prompts/2i-de-morgan_prompt_FB15k_test_2.csv" \
                    "../data_optimized/prompts/2u-de-morgan_prompt_FB15k_valid_2.csv" \
                    "../data_optimized/prompts/2u-de-morgan_prompt_FB15k_test_2.csv" \
                    
                    
    do
        wc -l $filename
        time python consistency_check_complex_queries_batch_execution.py \
        $filename \
        --nrows  $nrows \
        --chunksize $chunksize \
        --num_gpus $num_gpus \
        --model_name $model_name \
        --save_root ${save_root}
    done

done
