dataset="FB15k"
# dataset="NELL"
# dataset="wiki"
store_path="../data_optimized"
nrows=30000
# # 1c p, \neg p
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/1c_data_preprocessed_${dataset}_train.csv --bfs_depth 2 --nrows $nrows
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/1c_data_preprocessed_${dataset}_test.csv --bfs_depth 2 --nrows $nrows
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/1c_data_preprocessed_${dataset}_valid.csv --bfs_depth 2 --nrows $nrows

# # 2i p \wedge q
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/2i_data_preprocessed_${dataset}_train.csv --bfs_depth 2 --nrows $nrows
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/2i_data_preprocessed_${dataset}_test.csv --bfs_depth 2 --nrows $nrows
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/2i_data_preprocessed_${dataset}_valid.csv --bfs_depth 2 --nrows $nrows

# 2u p \vee q
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/2u_data_preprocessed_${dataset}_test.csv --bfs_depth 2 --nrows $nrows
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/2u_data_preprocessed_${dataset}_valid.csv --bfs_depth 2 --nrows $nrows

# # 1u2i p \vee (q \wedge r)
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/1u2i_data_preprocessed_${dataset}_test.csv --bfs_depth 2 --nrows $nrows
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/1u2i_data_preprocessed_${dataset}_valid.csv --bfs_depth 2 --nrows $nrows

# # # 1i2u (p \wedge q) \vee r
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/1i2u_data_preprocessed_${dataset}_test.csv --bfs_depth 2 --nrows $nrows
time python data_generation.py --dataset ${dataset} --store_path ${store_path} ${store_path}/1i2u_data_preprocessed_${dataset}_valid.csv --bfs_depth 2 --nrows $nrows


