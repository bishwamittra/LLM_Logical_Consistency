rm consistency_check/final_results/summary_table_*.csv
for tag in "test" "valid"
do
    for query_type in simple_with_rules simple conjunctive disjunctive commutative_2u commutative_2i onei2u oneu2i
    do

        echo $query_type
        
        for result_dir in \
            consistency_check/final_results/result_2024_05_07_before_finetuning_temperature_0 \
            consistency_check/final_results/result_2024_05_11_before_finetuning_temperature_0 \
            consistency_check/final_results/result_2024_05_11_before_finetuning_temperature_0_NELL \
            consistency_check/final_results/result_2024_05_14_before_finetuning_temperature_0_llama13b_chat_hf_NELL \
            consistency_check/final_results/result_2024_05_15_before_finetuning_temperature_0_gemma-2b-it_FB15k \
            consistency_check/final_results/result_2024_05_15_before_finetuning_temperature_0_llama7b_chat_hf_wiki \
            consistency_check/final_results/result_2024_05_16_before_finetuning_temperature_0_llama13b_chat_hf_wiki \
            consistency_check/final_results/result_2024_05_17_before_finetuning_temperature_0_gemma-2b-it_NELL \
            consistency_check/final_results/result_2024_05_18_before_finetuning_temperature_0_gemma-2b-it_wiki \

        do
            ls $result_dir | wc -l
            python read_output.py \
            $result_dir \
            --${query_type} \
            --tag $tag 
        done


        for result_dir in \
            consistency_check/final_results/result_2024_05_08_after_finetuning_learning_rate_2e-5_without_subqueries_temperature_0  \
            consistency_check/final_results/result_2024_05_10_after_finetuning_learning_rate_2e-5_without_subqueries_temperature_0 \
            consistency_check/final_results/result_2024_05_12_after_finetuning_llama7b_chat_hf_learning_rate_2e-5_without_subqueries_temperature_0_NELL \
            consistency_check/final_results/result_2024_05_15_after_finetuning_llama7b_chat_hf_learning_rate_2e-5_without_subqueries_temperature_0_wiki \
            consistency_check/final_results/result_2024_05_13_after_finetuning_llama13b_chat_hf_learning_rate_2e-5_without_subqueries_temperature_0_FB15k \
            consistency_check/result_2024_05_16_after_finetuning_llama13b_chat_hf_learning_rate_2e-5_without_subqueries_temperature_0_NELL \
            consistency_check/final_results/result_2024_05_16_after_finetuning_llama13b_chat_hf_learning_rate_2e-5_without_subqueries_temperature_0_wiki \
            consistency_check/final_results/result_2024_05_17_after_finetuning_gemma-2b-it_learning_rate_2e-6_without_subqueries_temperature_0_FB15k \
            consistency_check/final_results/result_2024_05_17_after_finetuning_gemma-2b-it_learning_rate_2e-6_without_subqueries_temperature_0_NELL \
            consistency_check/final_results/result_2024_05_18_after_finetuning_gemma-2b-it_learning_rate_2e-5_without_subqueries_temperature_0_wiki \

        do 

            ls $result_dir | wc -l
            python read_output.py \
            $result_dir \
            --${query_type} \
            --is_checkpoint_result \
            --tag $tag 
        done
        
    done
done