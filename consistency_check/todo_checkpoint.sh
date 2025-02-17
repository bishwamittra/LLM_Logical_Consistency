# 1c, 2i, 2u combined
checkpoint_path="../fine-tuning/results_2024_05_11_05_49_learning_rate_2e-5_without_subqueries_2000_samples_per_query_llama13b"


num_gpus=2
chunksize=1000
nrows=10000
dataset="FB15k"
model_name="llama13b_chat_hf"
save_root="result_$(date +%Y_%m_%d)_after_finetuning_${model_name}_learning_rate_2e-5_without_subqueries_temperature_0_${dataset}"

for checkpoint in 501 1002 1503 2004 2505 3006 3507
do
    # use checkpoint
    mkdir -p result
    finetuned_model_path="${save_root}/finetuned_model_$(date +%Y%m%d_%H%M%S)_$RANDOM"

    # restore checkpoint
    python checkpoint_restore.py \
    --base_model_path ../fine-tuning/base_models/Llama-2-13b-chat-hf \
    --checkpoint_path $checkpoint_path/checkpoint-$checkpoint \
    $finetuned_model_path


    # simple query
    for filename in \
                    "../data_optimized/prompts/1c_prompt_${dataset}_test_2.csv" \
                    "../data_optimized/prompts/1c_prompt_${dataset}_valid_2.csv" \
                    # "../data_optimized/prompts/1c_prompt_${dataset}_test_2_with_rules.csv" \
                    # "../data_optimized/prompts/1c_prompt_${dataset}_valid_2_with_rules.csv"
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

    # complex queries
    for filename in \
                    "../data_optimized/prompts/2i_prompt_${dataset}_test_2.csv" \
                    "../data_optimized/prompts/2i_prompt_${dataset}_valid_2.csv" \
                    "../data_optimized/prompts/2u_prompt_${dataset}_test_2.csv" \
                    "../data_optimized/prompts/2u_prompt_${dataset}_valid_2.csv" \
                    "../data_optimized/prompts/1i2u_prompt_${dataset}_test_2.csv" \
                    "../data_optimized/prompts/1i2u_prompt_${dataset}_valid_2.csv" \
                    "../data_optimized/prompts/1u2i_prompt_${dataset}_test_2.csv" \
                    "../data_optimized/prompts/1u2i_prompt_${dataset}_valid_2.csv"
    do
        wc -l $filename
        time python consistency_check_complex_queries_batch_execution.py \
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


    # commutative property
    for filename in \
                    "../data_optimized/prompts/2i_prompt_commutative_property_${dataset}_test_2_20000.csv" \
                    "../data_optimized/prompts/2i_prompt_commutative_property_${dataset}_valid_2_20000.csv" \
                    "../data_optimized/prompts/2u_prompt_commutative_property_${dataset}_test_2_20000.csv" \
                    "../data_optimized/prompts/2u_prompt_commutative_property_${dataset}_valid_2_20000.csv"
    do
        wc -l $filename
        time python consistency_check_complex_queries_commutative_property_batch_execution.py \
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

    # remove finetuned model
    rm -rf $finetuned_model_path
done
