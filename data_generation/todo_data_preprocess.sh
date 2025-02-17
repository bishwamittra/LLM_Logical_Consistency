dataset="FB15k"
# dataset="NELL"
# dataset="wiki"
store_path="../data_optimized"
# # 1c
time python data_preprocess.py train --query_type 1c --dataset $dataset --store_path $store_path
time python data_preprocess.py test --query_type 1c --dataset $dataset --store_path $store_path
time python data_preprocess.py valid --query_type 1c --dataset $dataset --store_path $store_path

# # 2i
time python data_preprocess.py train --query_type 2i --dataset $dataset --store_path $store_path
time python data_preprocess.py test --query_type 2i --dataset $dataset --store_path $store_path
time python data_preprocess.py valid --query_type 2i --dataset $dataset --store_path $store_path

# # # 2u
time python data_preprocess.py test --query_type 2u --dataset $dataset --store_path $store_path
time python data_preprocess.py valid --query_type 2u --dataset $dataset --store_path $store_path


# Before running preprocessing for 1i2u and 1u2i, run "bash todo_prepare_intersection_union.sh" to generate the intersection and union files
bash todo_prepare_intersection_union.sh

# 1i2u
time python data_preprocess.py test --query_type 1i2u --dataset $dataset --store_path $store_path
time python data_preprocess.py valid --query_type 1i2u --dataset $dataset --store_path $store_path

# 1u2i
time python data_preprocess.py test --query_type 1u2i --dataset $dataset --store_path $store_path
time python data_preprocess.py valid --query_type 1u2i --dataset $dataset --store_path $store_path
