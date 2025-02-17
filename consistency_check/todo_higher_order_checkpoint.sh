# 1c, 2i, 2u combined
checkpoint_path="../fine-tuning/backup_fine_tuned_results/results_2024_05_07_17_36_learning_rate_2e-5_without_subqueries_2000_samples_per_query_llama7b"

# FEVER only
# checkpoint_path="../fine-tuning/results_2024_08_12_18_42_learning_rate_2e-5_without_subqueries_2000_samples_per_query_llama7b_FEVER_shuffled"

num_gpus=2
chunksize=1000
nrows=4000
dataset="FEVER"
model_name="llama7b_chat_hf"
save_root="result_$(date +%Y_%m_%d)_after_finetuning_${model_name}_learning_rate_2e-5_without_subqueries_temperature_0_${dataset}"
save_root_chromadb="result_$(date +%Y_%m_%d)_after_finetuning_${model_name}_learning_rate_2e-5_without_subqueries_temperature_0_${dataset}_chromadb"
save_root_chromadbpreprocessed="result_$(date +%Y_%m_%d)_after_finetuning_${model_name}_learning_rate_2e-5_without_subqueries_temperature_0_${dataset}_chromadbPreprocessed"
        


for checkpoint in 501 1002 1503 2004 2505 3006 3507 4008 4509 5010 5511 6012 6513 7014 7515 8016 8517 9018 9519 10020
# for checkpoint in 250 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000 4250 4500 4750 5000
do
    # use checkpoint
    finetuned_model_path="${save_root}/finetuned_model_$(date +%Y%m%d_%H%M%S)_$RANDOM"

    # restore checkpoint
    python checkpoint_restore.py \
    --base_model_path /NS/llm-1/nobackup/shared/huggingface_cache/hub/meta-llama/Llama-2-7b-chat-hf \
    --checkpoint_path $checkpoint_path/checkpoint-$checkpoint \
    $finetuned_model_path

    # simple query
    for filename in \
                "../data_optimized/prompts/2i-exists-forall_prompt_FB15k_test_2.csv" \
                "../data_optimized/prompts/2i-exists-forall_prompt_FB15k_valid_2.csv" 
                    
    do
        wc -l $filename
        time python consistency_check_negation_batch_execution.py \
        $filename \
        --use_checkpoint \
        --finetuned_model_path $finetuned_model_path \
        --checkpoint $checkpoint \
        --nrows  $nrows \
        --chunksize $chunksize \
        --num_gpus $num_gpus \
        --model_name $model_name \
        --save_root ${save_root}
    done


    # # simple query
    # for filename in \
    #     "../data_optimized/prompts/1c_prompt_${dataset}_test_2.csv" \
    #     "../data_optimized/prompts/1c_prompt_${dataset}_valid_2.csv" \
                    
    # do
    #     wc -l $filename
    #     time python consistency_check_negation_batch_execution.py \
    #     $filename \
    #     --use_checkpoint \
    #     --finetuned_model_path $finetuned_model_path \
    #     --checkpoint $checkpoint \
    #     --nrows  $nrows \
    #     --chunksize $chunksize \
    #     --num_gpus $num_gpus \
    #     --model_name $model_name \
    #     --save_root ${save_root}
    # done

    # # simple query chromadb
    # for filename in \
    #     "../data_optimized/prompts/1c_prompt_chromadb${dataset}_test_2.csv" \
    #     "../data_optimized/prompts/1c_prompt_chromadb${dataset}_valid_2.csv" \
        
    # do
    #     wc -l $filename
    #     time python consistency_check_negation_batch_execution.py \
    #     $filename \
    #     --use_checkpoint \
    #     --finetuned_model_path $finetuned_model_path \
    #     --checkpoint $checkpoint \
    #     --nrows  $nrows \
    #     --chunksize $chunksize \
    #     --num_gpus $num_gpus \
    #     --model_name $model_name \
    #     --save_root ${save_root_chromadb}
    # done


    # # simple query chromadb
    # for filename in \
    #     "../data_optimized/prompts/1c_prompt_chromadbPreprocessed${dataset}_test_2.csv" \
    #     "../data_optimized/prompts/1c_prompt_chromadbPreprocessed${dataset}_valid_2.csv" \

    # do
    #     wc -l $filename
    #     time python consistency_check_negation_batch_execution.py \
    #     $filename \
    #     --use_checkpoint \
    #     --finetuned_model_path $finetuned_model_path \
    #     --checkpoint $checkpoint \
    #     --nrows  $nrows \
    #     --chunksize $chunksize \
    #     --num_gpus $num_gpus \
    #     --model_name $model_name \
    #     --save_root ${save_root_chromadbpreprocessed}
    # done

    
    # remove finetuned model
    echo $finetuned_model_path
    rm -rf $finetuned_model_path
done
