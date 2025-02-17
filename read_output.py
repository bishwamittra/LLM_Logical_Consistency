import os
import pandas as pd
import numpy as np
import argparse


# helper functions
def mean_std_df(df, group_columns, columns_to_agg):
    xdf = df.groupby(group_columns).agg({column : [np.mean, np.std] for column in columns_to_agg})
    xdf.columns = xdf.columns.map("_".join)
    return xdf.reset_index()

def print_summary(df, list_group, list_aggregate, show_std=False):
    df_summary = mean_std_df(df, list_group, list_aggregate)
    for column in list_aggregate:
        # df_summary[column] = df_summary.apply(lambda x: f"$ {x[column+'_mean']} \pm {x[column+'_std']} $", axis=1)
        # check nan
        if(show_std or column.startswith("time")):
            df_summary[column] = df_summary.apply(lambda x: f"\\textemdash" if not pd.notna(x[column+'_mean'])
                                    else f"$ {x[column+'_mean']:.2f} \pm {x[column+'_std']:.2f} $", axis=1)
        else:
            df_summary[column] = df_summary.apply(lambda x: f"\\textemdash" if not pd.notna(x[column+'_mean'])
                                    else f"$ {x[column+'_mean']:.2f} $", axis=1)
        # df_summary[column] = df_summary.apply(lambda x: f"$ {x[column+'_mean']} ({x[column+'_std']}) $", axis=1)

    return df_summary[list_group + list_aggregate]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_checkpoint_result", action="store_true", default=False, help="Whether the result is from checkpoint")
    parser.add_argument("--simple", action="store_true", default=False, help="Whether the result is from simple query")
    parser.add_argument("--simple_with_rules", action="store_true", default=False, help="Whether the result is from simple query")
    parser.add_argument("--commutative_2u", action="store_true", default=False, help="Whether the result is from commutative property of 2u")
    parser.add_argument("--commutative_2i", action="store_true", default=False, help="Whether the result is from commutative property of 2i")
    parser.add_argument("--oneu2i", action="store_true", default=False, help="Whether the result is from 1u2i query")
    parser.add_argument("--onei2u", action="store_true", default=False, help="Whether the result is from 1i2u query")
    parser.add_argument("--conjunctive", action="store_true", default=False, help="Whether the result is from conjunctive query")
    parser.add_argument("--disjunctive", action="store_true", default=False, help="Whether the result is from disjunctive query")
    parser.add_argument("--tag", type=str, default="test", choices=['test', 'train', 'valid'], help="Tag for the result")
    parser.add_argument("result_dir", type=str, help="Result directory")
    args = parser.parse_args()

    assert args.simple + args.commutative_2u + args.commutative_2i + args.conjunctive + args.disjunctive + args.simple_with_rules + args.onei2u + args.oneu2i <= 1

    is_checkpoint_result = args.is_checkpoint_result

    commutative_2u = args.commutative_2u
    commutative_2i = args.commutative_2i
    conjunctive = args.conjunctive
    disjunctive = args.disjunctive
    simple = args.simple
    simple_with_rules = args.simple_with_rules

    # if(is_checkpoint_result):
    #     result_dir = "consistency_check/result_2024_05_08_after_finetuning_learning_rate_2e-5_without_subqueries_temperature_0"
    # else:
    #     result_dir = "consistency_check/final_results/result_2024_05_07_before_finetuning_temperature_0"
    result_dir = args.result_dir


    # get corresponding exp_list
    exp_list = {}
    for folder in os.listdir(result_dir):
        if(folder == "exp_seq.txt"):
            continue
        for result_file in os.listdir(f"{result_dir}/{folder}"):
            if(result_file.endswith(".csv") and args.tag in result_file):
                if(args.simple and "1c" in result_file and "with_rules" not in result_file) or \
                    (args.simple_with_rules and "1c" in result_file and "with_rules" in result_file) or \
                    (args.oneu2i and result_file.startswith("1u2i")) or \
                    (args.onei2u and result_file.startswith("1i2u")) or \
                    (args.commutative_2u and result_file.startswith("2u") and "commutative_property" in result_file) or \
                    (args.commutative_2i and result_file.startswith("2i") and "commutative_property" in result_file) or \
                    (args.conjunctive and result_file.startswith("2i") and "commutative_property" not in result_file) or \
                    (args.disjunctive and result_file.startswith("2u") and "commutative_property" not in result_file) \
                :                    
                    if(is_checkpoint_result):
                        exp_list[int(folder.split("checkpoint_")[1])] = int(folder.split("_")[1])
                    else:
                        exp_list['base'] = int(folder.split("_")[1])

    exp_list = sorted(list(exp_list.values()))
    print(exp_list)
    if(len(exp_list) == 0):
        print("No corresponding result")
        exit()


    checkpoints = []
    if(is_checkpoint_result):
        for i, exp in enumerate(exp_list):
            for folder in os.listdir(result_dir):
                if(simple or simple_with_rules):
                    if(folder.startswith(f"exp_{exp}_negation_consistency_on_checkpoint")):
                        checkpoints.append(int(folder.split("_")[-1]))
                else:
                    if(folder.startswith(f"exp_{exp}_complex_query_consistency_on_checkpoint")):
                        checkpoints.append(int(folder.split("_")[-1]))

    print(checkpoints)
    df = pd.DataFrame()
    for i, exp in enumerate(exp_list):
        if(is_checkpoint_result):
            if(simple or simple_with_rules):
                dir = f"{result_dir}/exp_{exp}_negation_consistency_on_checkpoint_{checkpoints[i]}"
            else:
                dir = f"{result_dir}/exp_{exp}_complex_query_consistency_on_checkpoint_{checkpoints[i]}"
        else:
            if(simple or simple_with_rules):
                dir = f"{result_dir}/exp_{exp}_negation_consistency"
            else:
                dir = f"{result_dir}/exp_{exp}_complex_query_consistency"


        for file in os.listdir(dir):
            if(file.endswith(".csv")):
                model_name = file.split("_response_")[0].split("_")[1:]
                model_name = "_".join(model_name)
                if("_response_commutative_property_" in file):
                    dataset_name = file.split("_response_commutative_property_")[1].split(".")[0].split("_")[0]
                else:
                    dataset_name = file.split("_response_")[1].split(".")[0].split("_")[0]
                
                df_ = pd.read_csv(f"{dir}/{file}", lineterminator='\n')
                print(model_name, file, df_.shape)
                df_['model_name'] = model_name
                df_['dataset'] = dataset_name
                if(is_checkpoint_result):
                    df_['checkpoint'] = checkpoints[i]
                else:
                    df_['checkpoint'] = False
                dropped_index = {}
                for column in df_.columns:
                    if(not column.startswith("time")):
                        continue
                    for i in range(df_.shape[0]):
                        if(df_.iloc[i][column] in ["True", "False"]):
                            dropped_index[i] = True
                dropped_index = list(dropped_index.keys())
                if(len(dropped_index) > 0):
                    print(f"Dropped index: {dropped_index}")
                    df_ = df_.drop(dropped_index)
                df = pd.concat([df, df_])


    num_subqueries = 0
    for column in df.columns:
        if("correct_subquery_" in column):
            num_subqueries += 1


    for column in df.columns:
        if("prompt" in column or "raw_response" in column):
            df[column] = df[column].apply(str)
            df[column] = df[column].apply(lambda x: x.replace('[NEWLINE]', '\n'))

        if(column.startswith("time")):
            df[column] = df[column].apply(float)

        if(column.startswith("correct") or
           column.startswith("logically_consistent") or
           column.startswith("use_context") or
           column.startswith("response") or
           column.startswith("ground_truth")
           ):
            df[column] = df[column].apply(bool)

    print(f"Num of subqueries: {num_subqueries}")
    print(df.shape)
    df.head()


    # a combined table



    good_name_model = {
        "llama7b_chat_hf" : "Llama$2$-$7$B",
        "llama13b_chat_hf": "Llama$2$-$13$B",
        "gemma-2b-it": "Gemma-$2$B",
    }
    good_name_dataset = {
        "FB15k": "FreebaseLFC",
        "wiki": "WikiLFC",
        "NELL": "NELLLFC"
    }
    df['model_name'] = df['model_name'].apply(lambda x: good_name_model[x] if x in good_name_model else x)
    df['dataset'] = df['dataset'].apply(lambda x: good_name_dataset[x] if x in good_name_dataset else x)

    assert "time_total" not in df.columns
    df['time_total'] = df['time_bfs'] + df['time_context'] + df['time_response_base_query']
    list_efficiency = ['time_bfs','time_context', 'time_flipped_entity_find', 'time_response_base_query', 'time_total']
    if(simple or simple_with_rules):
        # combine base and negation query accuracy
        df_melt = pd.melt(df, 
                        id_vars=['model_name', 'dataset', 'use_context', 'checkpoint', 'logically_consistent'] + list_efficiency, 
                        value_vars=['correct_base_query', 'correct_negation_query'],
                        var_name='query_type',
                        value_name='accuracy')
        
        list_performance = ['accuracy', 'logically_consistent']
    else:
        list_performance = ['correct_base_query', 'logically_consistent']

    list_performance += list_efficiency
    list_group = ['model_name', 'dataset'] + ['checkpoint']
    if(simple or simple_with_rules):
        df_summary_with_checkpoint = print_summary(df_melt[df_melt['use_context'] == True], list_group, list_performance)
    else:
        df_summary_with_checkpoint = print_summary(df[df['use_context'] == True], list_group, list_performance)


    final_table = {
        "Model": [],
        "Dataset": [],
        "Fact": [],
        "Fine-tune": [],
        "Checkpoint": [],
        "Accuracy": [],
        "Logical Consistency": [],
        "Time BFS": [],
        "Time Context": [],
        "Time Response Base Query": [],
        "Time Flipped Entity Find": [],
        "Time Total": [],
    }

    group_list = ['checkpoint']

    for checkpoint, df_summary in df_summary_with_checkpoint.groupby(group_list):
        
        final_table["Model"].append(df_summary["model_name"].item())
        final_table['Dataset'].append(df_summary['dataset'].item())
        final_table["Checkpoint"].append(checkpoint[0])
        if(simple or simple_with_rules):
            final_table["Accuracy"].append(df_summary["accuracy"].item())
        else:
            final_table["Accuracy"].append(df_summary["correct_base_query"].item())
        final_table["Logical Consistency"].append(df_summary["logically_consistent"].item())
        final_table["Time BFS"].append(df_summary["time_bfs"].item())
        final_table["Time Context"].append(df_summary["time_context"].item())
        final_table["Time Response Base Query"].append(df_summary["time_response_base_query"].item())
        final_table["Time Flipped Entity Find"].append(df_summary["time_flipped_entity_find"].item())
        final_table["Time Total"].append(df_summary["time_total"].item())
        if(disjunctive):
            final_table["Fact"].append(" $ p \\vee q $ ")
        elif(conjunctive):
            final_table["Fact"].append(" $ p \\wedge q $ ")
        elif(commutative_2i):
            final_table["Fact"].append(" $ p \\wedge q  \\leftrightarrow q \\wedge p $ ")
        elif(commutative_2u):
            final_table["Fact"].append(" $ p \\vee q  \\leftrightarrow q \\vee p $ ")
        elif(simple):
            final_table["Fact"].append(" $ p, \\neg p $ ")
        elif(simple_with_rules):
            final_table["Fact"].append(" $ p, \\neg p $  with rules")
        elif(args.oneu2i):
            final_table["Fact"].append(" $ p \\vee (q \\wedge r) $ ")
        elif(args.onei2u):
            final_table["Fact"].append(" $ p \\wedge (q \\vee r) $ ")
        else:
            raise ValueError("Unknown query type")
        if(is_checkpoint_result):
            final_table["Fine-tune"].append("After Fine-tuning")
        else:
            final_table["Fine-tune"].append("Before Fine-tuning")

    final_table = pd.DataFrame(final_table)
    final_table


    # store
    if not os.path.isfile(f"consistency_check/final_results/summary_table_{args.tag}.csv"):
        final_table.to_csv(f"consistency_check/final_results/summary_table_{args.tag}.csv", index=False)
    else:
        final_table.to_csv(f"consistency_check/final_results/summary_table_{args.tag}.csv", mode='a', header=False, index=False)



    # detailed results on base and subquery assignments. Do not track time information
    max_subqueries = 5
    if(simple or simple_with_rules):

        df_base_query = df[['model_name', 'dataset', 'use_context', 'checkpoint', 'logically_consistent', 'ground_truth_base_query', 'correct_base_query']]
        df_base_query = df_base_query.rename(columns={'correct_base_query': 'accuracy'})

        df_negation_query = df[['model_name', 'dataset', 'use_context', 'checkpoint', 'logically_consistent', 'ground_truth_negation_query', 'correct_negation_query']]
        df_negation_query = df_negation_query.rename(
                columns={'correct_negation_query': 'accuracy',
                 'ground_truth_negation_query': 'ground_truth_base_query'})
        
        df_combined = pd.concat([df_base_query, df_negation_query])
        list_performance = ['accuracy', 'logically_consistent']
    else:
        list_performance = ['correct_base_query', 'logically_consistent']

    list_group = ['model_name', 'dataset'] + ['checkpoint']
    if(simple or simple_with_rules):
        list_group += ['ground_truth_base_query']
        df_summary_with_checkpoint = print_summary(df_combined[df_combined['use_context'] == True], list_group, list_performance)
    else:
        list_group += ['ground_truth_base_query'] + [f'ground_truth_subquery_{i+1}' for i in range(num_subqueries)]
        df_summary_with_checkpoint = print_summary(df[df['use_context'] == True], list_group, list_performance)


    final_table_detail = {
        "Model": [],
        "Dataset": [],
        "Fact": [],
        "Fine-tune": [],
        "Checkpoint": [],
        "Ground Truth Base Query": [],
        "Accuracy": [],
        "Logical Consistency": [],
    }
    for i in range(max_subqueries):
        final_table_detail[f"Ground Truth Subquery {i+1}"] = []
    
    group_list = ['checkpoint']
    if(simple or simple_with_rules):
        group_list += ['ground_truth_base_query']
    else:
        group_list += ['ground_truth_base_query'] + [f'ground_truth_subquery_{i+1}' for i in range(num_subqueries)]
    
    for checkpoint, df_summary in df_summary_with_checkpoint.groupby(group_list):
        final_table_detail["Model"].append(df_summary["model_name"].item())
        final_table_detail['Dataset'].append(df_summary['dataset'].item())
        final_table_detail["Checkpoint"].append(checkpoint[0])
        if(simple or simple_with_rules):
            final_table_detail["Accuracy"].append(df_summary["accuracy"].item())
        else:
            final_table_detail["Accuracy"].append(df_summary["correct_base_query"].item())
        final_table_detail["Logical Consistency"].append(df_summary["logically_consistent"].item())
        if(disjunctive):
            final_table_detail["Fact"].append(" $ p \\vee q $ ")
        elif(conjunctive):
            final_table_detail["Fact"].append(" $ p \\wedge q $ ")
        elif(commutative_2i):
            final_table_detail["Fact"].append(" $ p \\wedge q  \\leftrightarrow q \\wedge p $ ")
        elif(commutative_2u):
            final_table_detail["Fact"].append(" $ p \\vee q  \\leftrightarrow q \\vee p $ ")
        elif(simple):
            final_table_detail["Fact"].append(" $ p, \\neg p $ ")
        elif(simple_with_rules):
            final_table_detail["Fact"].append(" $ p, \\neg p $  with rules")
        elif(args.oneu2i):
            final_table_detail["Fact"].append(" $ p \\vee (q \\wedge r) $ ")
        elif(args.onei2u):
            final_table_detail["Fact"].append(" $ p \\wedge (q \\vee r) $ ")
        else:
            raise ValueError("Unknown query type")
        if(is_checkpoint_result):
            final_table_detail["Fine-tune"].append("After Fine-tuning")
        else:
            final_table_detail["Fine-tune"].append("Before Fine-tuning")
        for i in range(max_subqueries):
            if(i >= num_subqueries):
                final_table_detail[f"Ground Truth Subquery {i+1}"].append(None)
            else:
                final_table_detail[f"Ground Truth Subquery {i+1}"].append(df_summary[f'ground_truth_subquery_{i+1}'].item())
        final_table_detail["Ground Truth Base Query"].append(df_summary['ground_truth_base_query'].item())

    final_table_detail = pd.DataFrame(final_table_detail)
    final_table_detail

    # store
    if not os.path.isfile(f"consistency_check/final_results/summary_table_detail_{args.tag}.csv"):
        final_table_detail.to_csv(f"consistency_check/final_results/summary_table_detail_{args.tag}.csv", index=False)
    else:
        final_table_detail.to_csv(f"consistency_check/final_results/summary_table_detail_{args.tag}.csv", mode='a', header=False, index=False)

