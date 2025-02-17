num_gpus=2
chunksize=500
nrows=10000
model_name="llama7b_chat_hf"
save_root="result_$(date +%Y_%m_%d)_before_finetuning_temperature_0"



# simple query
for filename in \
                "../data_optimized/prompts/1c_prompt_FB15k_test_2.csv" \
                "../data_optimized/prompts/1c_prompt_FB15k_valid_2.csv" \
                "../data_optimized/prompts/1c_prompt_FB15k_test_2_with_rules.csv" \
                "../data_optimized/prompts/1c_prompt_FB15k_valid_2_with_rules.csv"
do
    wc -l $filename
    time python consistency_check_negation_batch_execution.py \
    $filename \
    --nrows  $nrows \
    --chunksize $chunksize \
    --num_gpus $num_gpus \
    --model_name $model_name \
    --save_root ${save_root}
done

# complex queries
for filename in \
                "../data_optimized/prompts/2i_prompt_FB15k_test_2.csv" \
                "../data_optimized/prompts/2i_prompt_FB15k_valid_2.csv" \
                "../data_optimized/prompts/2u_prompt_FB15k_test_2.csv" \
                "../data_optimized/prompts/2u_prompt_FB15k_valid_2.csv" \
                "../data_optimized/prompts/1i2u_prompt_FB15k_test_2.csv" \
                "../data_optimized/prompts/1i2u_prompt_FB15k_valid_2.csv" \
                "../data_optimized/prompts/1u2i_prompt_FB15k_test_2.csv" \
                "../data_optimized/prompts/1u2i_prompt_FB15k_valid_2.csv"
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


# commutative property
for filename in \
                "../data_optimized/prompts/2i_prompt_commutative_property_FB15k_test_2_20000.csv" \
                "../data_optimized/prompts/2i_prompt_commutative_property_FB15k_valid_2_20000.csv" \
                "../data_optimized/prompts/2u_prompt_commutative_property_FB15k_test_2_20000.csv" \
                "../data_optimized/prompts/2u_prompt_commutative_property_FB15k_valid_2_20000.csv"
do
    wc -l $filename
    time python consistency_check_complex_queries_commutative_property_batch_execution.py \
    $filename \
    --nrows  $nrows \
    --chunksize $chunksize \
    --num_gpus $num_gpus \
    --model_name $model_name \
    --save_root ${save_root}
done



